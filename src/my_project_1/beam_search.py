"""Educational fixed-bank beam search; candidate generation, not intent checking."""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from .generate import generate_reply
from .model import TinyDecoderLM
from .tokenization import encode_parts, file_hash, load_frozen, read_rows
from .train import _save_json


def rank_score(log_probability, length, alpha):
    return log_probability / max(1, length) ** alpha


@torch.no_grad()
def beam_search_reply(model, tokenizer, model_config, instruction, *, beam_width=4,
                      num_return_sequences=4, max_new_tokens=320, length_penalty=0.7):
    """Top B prefix bank, including finished paths (which occupy slots).

    At each step, expand each active path to its B best allowed tokens; combine
    with carried finished paths, rank by sum(original log p)/length**alpha,
    keep B. EOS counts in length and budget. No padding, sampling, reranker or
    guaranteed global optimum; prefix length normalization is a search heuristic.
    Stable ties use token-ID lexicographic order. B=1 is greedy, including ties.
    """
    for name, value in [('beam_width', beam_width), ('num_return_sequences', num_return_sequences),
                        ('max_new_tokens', max_new_tokens)]:
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    if beam_width > 16 or num_return_sequences > beam_width:
        raise ValueError('Require return_count <= beam_width <= 16')
    if isinstance(length_penalty, bool) or not isinstance(length_penalty, (int, float)) or not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError('length_penalty must be finite and nonnegative')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction must be nonempty')
    prompt = encode_parts(tokenizer, instruction, '')[2][:-1]
    budget = model_config['max_length']
    if len(prompt) >= budget:
        raise ValueError('Prompt leaves no room for generation')
    device = next(model.parameters()).device
    eos = model_config['eos_token_id']
    blocked = [model_config[k + '_token_id'] for k in ('pad', 'bos', 'user', 'assistant')]
    bank = [{'ids': [], 'log_probability': 0.0, 'stop_reason': None}]
    trace = []
    was_training = model.training
    model.eval()
    try:
        for step in range(1, min(max_new_tokens, budget - len(prompt)) + 1):
            active = [(i, b) for i, b in enumerate(bank) if b['stop_reason'] is None]
            if not active:
                break
            ids = torch.tensor([prompt + b['ids'] for _, b in active], dtype=torch.long, device=device)
            logits = model(ids, attention_mask=torch.ones_like(ids))['logits'][:, -1]
            if not logits.isfinite().all():
                raise ValueError('Non-finite beam logits')
            # Score original model probabilities; suppress choices without renormalizing.
            logp = logits.log_softmax(dim=-1)
            logp[:, blocked] = -float('inf')
            proposals = [{**b, 'parent_rank': i + 1, 'token': None, 'token_log_probability': None,
                          'carried': True} for i, b in enumerate(bank) if b['stop_reason'] is not None]
            for j, (i, b) in enumerate(active):
                choices = torch.argsort(logp[j], descending=True, stable=True)[:beam_width].tolist()
                for token in choices:
                    value = float(logp[j, token])
                    if not math.isfinite(value):
                        continue
                    new_ids = b['ids'] + [token]
                    reason = ('eos' if token == eos else
                              'context_limit' if len(prompt) + len(new_ids) >= budget else
                              'max_new_tokens' if step == max_new_tokens else None)
                    proposals.append({'ids': new_ids, 'log_probability': b['log_probability'] + value,
                                      'stop_reason': reason, 'parent_rank': i + 1, 'token': token,
                                      'token_log_probability': value, 'carried': False})
            proposals.sort(key=lambda b: (-rank_score(b['log_probability'], len(b['ids']), length_penalty), tuple(b['ids'])))
            if not proposals:
                raise ValueError('No allowed finite beam candidates')
            bank = proposals[:beam_width]
            trace.append({'step': step, 'expanded_paths': len(active), 'proposals': len(proposals),
                          'kept': [{'rank': k + 1, 'parent_rank': b['parent_rank'], 'token': b['token'],
                                    'token_log_probability': b['token_log_probability'], 'carried': b['carried'],
                                    'length': len(b['ids']), 'log_probability': b['log_probability'],
                                    'score': rank_score(b['log_probability'], len(b['ids']), length_penalty),
                                    'stop_reason': b['stop_reason']} for k, b in enumerate(bank)]})
    finally:
        model.train(was_training)
    candidates = [{'rank': k + 1, 'response': tokenizer.decode(b['ids'], skip_special_tokens=True),
                   'generated_token_ids': b['ids'], 'generated_tokens': len(b['ids']),
                   'stop_reason': b['stop_reason'], 'log_probability': b['log_probability'],
                   'score': rank_score(b['log_probability'], len(b['ids']), length_penalty)}
                  for k, b in enumerate(bank[:num_return_sequences])]
    return {'prompt_tokens': len(prompt), 'candidates': candidates, 'trace': trace,
            'unique_response_count': len({c['response'] for c in candidates}),
            'ranking': 'sum original log probabilities / generated_length**length_penalty; EOS included'}


def run_demo(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    checkpoint_path = Path(config['checkpoint'])
    if file_hash(checkpoint_path) != config['checkpoint_sha256']:
        raise ValueError('Fifth checkpoint hash changed')
    if file_hash(config['selection']) != config['selection_sha256'] or file_hash(config['dev']) != config['dev_sha256']:
        raise ValueError('Frozen dev inputs changed')
    selection = json.loads(Path(config['selection']).read_text(encoding='utf-8'))['samples']
    by_id = {r['id']: r for r in read_rows(config['dev'])}
    # Fixed first selected example of each configured intent, before inference.
    chosen = [next(s for s in selection if s['intent'] == intent) for intent in config['demo_intents']]
    tokenizer, tc = load_frozen(config['tokenizer_artifact'])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config['cpu_threads'])
    try:
        torch.manual_seed(config['seed'])
        data = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        mc = data['model_config']
        if mc['tokenizer_revision'] != tc['tokenizer_sha256']:
            raise ValueError('Tokenizer mismatch')
        model = TinyDecoderLM.from_config(mc)
        model.load_state_dict(data['model_state_dict'], strict=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        out = Path(config['output_dir']) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex)
        out.mkdir(parents=True, exist_ok=False)
        dependencies = [config_path, checkpoint_path, Path(config['selection']), Path(config['dev']),
                        Path(__file__), Path('src/my_project_1/generate.py'), Path('src/my_project_1/model.py')]
        _save_json(out / 'config.json', {**config, 'model_config': mc, 'resolved_device': 'cpu',
                   'dependencies_sha256': {str(p): file_hash(p) for p in dependencies},
                   'inputs': [{**s, 'instruction': by_id[s['id']]['instruction'],
                               'input_ids': encode_parts(tokenizer, by_id[s['id']]['instruction'], '')[2][:-1]} for s in chosen]})
        with (out / 'candidates.jsonl').open('x', encoding='utf-8') as stream:
            for s in chosen:
                row = by_id[s['id']]
                print('\nQuestion: ' + row['instruction'], flush=True)
                greedy = generate_reply(model, tokenizer, mc, row['instruction'], max_new_tokens=config['max_new_tokens'])
                result = beam_search_reply(model, tokenizer, mc, row['instruction'], beam_width=config['beam_width'],
                          num_return_sequences=config['num_return_sequences'], max_new_tokens=config['max_new_tokens'],
                          length_penalty=config['length_penalty'])
                record = {'id': row['id'], 'instruction': row['instruction'], 'greedy': greedy, 'beam': result,
                          'candidate_quality': {'status': 'not_scored', 'greedy_quality': None, 'beam_top1_quality': None,
                                                'acceptable_candidate_present': None},
                          'checker_selection': {'status': 'not_implemented', 'selected_rank': None,
                                                'selection_correct': None, 'gain_vs_beam_top1': None}}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush()
                print('Greedy: ' + greedy['response'], flush=True)
                for c in result['candidates']:
                    print(f"Beam #{c['rank']} score={c['score']:.4f} logP={c['log_probability']:.4f} stop={c['stop_reason']}\n{c['response']}", flush=True)
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        deps = json.loads((out / 'config.json').read_text(encoding='utf-8'))['dependencies_sha256']
        assert all(file_hash(p) == h for p, h in deps.items())
        _save_json(out / 'summary.json', {'status': 'completed', 'examples': len(chosen), 'weights_unchanged': True,
                   'optimizer_updates': 0, 'quality_improvement': None, 'checker_effectiveness': None,
                   'scope': 'two fixed learning examples; not 135-dev evaluation, not evidence of quality gains',
                   'candidates_sha256': file_hash(out / 'candidates.jsonl')})
        print('\nSaved: ' + str(out.resolve()), flush=True)
        return out
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/beam-search-fifth.json'))
    args = parser.parse_args()
    run_demo(args.config)


if __name__ == '__main__':
    main()
