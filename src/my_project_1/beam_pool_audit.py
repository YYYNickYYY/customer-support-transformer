"""Frozen 12-question candidate-pool audit, not training or checker selection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import torch

from .beam_search import beam_search_reply
from .expanded_dev import read, save
from .generate import generate_reply
from .model import TinyDecoderLM
from .tokenization import encode_parts, file_hash, load_frozen, read_rows

OUT = Path('outputs/beam-pool-audit-v1-20261007')
INTENTS = ('cancel_order', 'change_order', 'create_account', 'delete_account', 'get_refund', 'check_refund_policy')
PROTOCOL = Path('docs/beam-pool-audit-v1-protocol.md')


def choose(selection, rows):
    """Instruction-only eligibility, salted ID hash order; no model outputs."""
    by_id = {r['id']: r for r in rows}
    chosen = []
    for intent in INTENTS:
        pool = [s for s in selection['samples'] if s['intent'] == intent]
        if intent == 'cancel_order':
            pool = [s for s in pool if re.search(r'cancel', by_id[s['id']]['instruction'], re.I)]
        elif intent == 'change_order':
            pool = [s for s in pool if re.search(r'\b(remov\w*|delet\w*)\b', by_id[s['id']]['instruction'], re.I)]
        if len(pool) < 2:
            raise ValueError('Too few instruction-eligible questions: ' + intent)
        pool.sort(key=lambda s: (hashlib.sha256(('beam-pool-v1:42:' + s['id']).encode()).hexdigest(), s['id']))
        chosen.extend({**s, 'instruction': by_id[s['id']]['instruction']} for s in pool[:2])
    return chosen


def prepare():
    config = read('configs/beam-search-fifth.json')
    assert file_hash(config['checkpoint']) == config['checkpoint_sha256']
    assert file_hash(config['selection']) == config['selection_sha256']
    assert file_hash(config['dev']) == config['dev_sha256']
    chosen = choose(read(config['selection']), list(read_rows(config['dev'])))
    tokenizer, tc = load_frozen(config['tokenizer_artifact'])
    for s in chosen:
        s['input_ids'] = encode_parts(tokenizer, s['instruction'], '')[2][:-1]
    deps = ['configs/beam-search-fifth.json', config['checkpoint'], config['selection'], config['dev'],
            str(PROTOCOL), str(Path(__file__)), 'src/my_project_1/beam_search.py', 'src/my_project_1/generate.py',
            'src/my_project_1/model.py', 'src/my_project_1/attention.py', 'src/my_project_1/transformer.py',
            'src/my_project_1/tokenization.py', 'tests/test_beam_pool_audit.py',
            'docs/evaluation-rubric-v1.md', 'docs/evaluation-risk-rubric-v1.1.md',
            config['tokenizer_artifact'] + '/tokenizer.json', config['tokenizer_artifact'] + '/config.json']
    plan = {'version': 'beam-pool-audit-v1', 'samples': chosen, 'generation': config,
            'tokenizer_revision': tc['tokenizer_sha256'], 'torch_version': str(torch.__version__),
            'dependencies_sha256': {p: file_hash(p) for p in deps}}
    save(OUT / 'freeze.json', plan)
    print(json.dumps({'frozen': True, 'examples': len(chosen), 'ids': [s['id'] for s in chosen]}, ensure_ascii=False), flush=True)


def verify():
    plan = read(OUT / 'freeze.json')
    for p, h in plan['dependencies_sha256'].items():
        if file_hash(p) != h:
            raise ValueError('Frozen dependency changed: ' + p)
    if str(torch.__version__) != plan['torch_version']:
        raise ValueError('Runtime changed')
    return plan


def generate():
    plan = verify()
    config = plan['generation']
    path = OUT / 'candidates.jsonl'
    prior = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if len(prior) > 12:
        raise ValueError('Too many records')
    for saved, s in zip(prior, plan['samples']):
        if any(saved[k] != s[k] for k in ('id', 'intent', 'instruction', 'input_ids')):
            raise ValueError('Saved prefix mismatch')
    if len(prior) == 12:
        assert read(OUT / 'generation-summary.json')['candidates_sha256'] == file_hash(path)
        print('Completed generation verified; no new inference.')
        return
    tokenizer, _ = load_frozen(config['tokenizer_artifact'])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config['cpu_threads'])
    try:
        torch.manual_seed(config['seed'])
        data = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
        mc = data['model_config']
        model = TinyDecoderLM.from_config(mc)
        model.load_state_dict(data['model_state_dict'], strict=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with path.open('a' if prior else 'x', encoding='utf-8') as stream:
            for i, s in enumerate(plan['samples'][len(prior):], len(prior) + 1):
                greedy = generate_reply(model, tokenizer, mc, s['instruction'], max_new_tokens=config['max_new_tokens'])
                beam = beam_search_reply(model, tokenizer, mc, s['instruction'], beam_width=config['beam_width'],
                        num_return_sequences=config['num_return_sequences'], max_new_tokens=config['max_new_tokens'],
                        length_penalty=config['length_penalty'])
                assert len(beam['candidates']) == 4
                record = {**s, 'greedy': greedy, 'beam': beam,
                          'checker_selection': {'status': 'not_implemented', 'selected_rank': None, 'effectiveness': None}}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush()
                prior.append(record)
                print(f'{i}/12 {s["intent"]}: greedy={greedy["stop_reason"]}; beam={[c["stop_reason"] for c in beam["candidates"]]}', flush=True)
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        verify()
        save(OUT / 'generation-summary.json', {'examples': 12, 'beam_candidates': 48,
             'weights_unchanged': True, 'optimizer_updates': 0, 'candidates_sha256': file_hash(path),
             'greedy_stops': dict(Counter(r['greedy']['stop_reason'] for r in prior)),
             'beam_stops': dict(Counter(c['stop_reason'] for r in prior for c in r['beam']['candidates']))})
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'generate', 'check'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'generate':
        generate()
    else:
        verify(); print('Frozen audit verified; no inference')


if __name__ == '__main__':
    main()
