"""Freeze deterministic case selection, then trace the existing fifth checkpoint."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
import tokenizers

from my_project_1.expanded_dev import read, save
from my_project_1.generation_diagnostic import trace_reply, locate_quote
from my_project_1.model import TinyDecoderLM
from my_project_1.tokenization import file_hash, load_frozen

OUT = Path('outputs/fifth-generation-diagnostic-20261006-v1')
SOURCE = Path('outputs/dev-expanded-fifth-20261005')
RATINGS = Path('outputs/repetition-review-20261006-v1/ai-expert-ratings-repetition-revised.json')


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def choose_cases(replies, ratings):
    by_id = {r['id']: r for r in replies}
    if len(by_id) != len(replies):
        raise ValueError('Duplicate reply ID')
    fifth = {r['id']: r for r in ratings['ratings'] if r['model'] == 'fifth'}
    paired = ratings['pairs']
    chosen = []
    seen = set()
    def add(pool, stratum, limit=None):
        pool = sorted(pool, key=lambda k: (hashlib.sha256(
            f'fifth-generation-diagnostic-v1:42:{stratum}:{k}'.encode()).hexdigest(), k))
        used = 0
        for k in pool:
            if k in seen:
                continue
            if limit is not None and used == limit:
                break
            if k not in by_id:
                raise ValueError('Case missing from frozen replies')
            chosen.append({'id': k, 'stratum': stratum})
            seen.add(k)
            used += 1
        if limit is not None and used != limit:
            raise ValueError('Insufficient diagnostic stratum')
    focus = [k for k in by_id if k.rsplit(':', 1)[-1] in ('16355', '26320', '26438')]
    add(focus, 'refund-guarantee', 3)
    add([r['id'] for r in paired if r['fourth'] == 0 and r['fifth'] == 1], 'new-repeat', 2)
    add([r['id'] for r in paired if r['fourth'] == 1 and r['fifth'] == 1], 'persistent-repeat', 2)
    add([k for k, r in fifth.items() if not r['repetition'] and not r['unsupported_claim']], 'control', 3)
    add([k for k, r in by_id.items() if r['stop_reason'] != 'eos'], 'non-eos')
    return [{**r, 'instruction': by_id[r['id']]['instruction'], 'intent': by_id[r['id']]['intent']}
            for r in chosen]


def verify_hashes(mapping):
    for p, h in mapping.items():
        if file_hash(p) != h:
            raise ValueError('Changed diagnostic source: ' + p)


def prepare():
    if OUT.exists():
        raise ValueError('Do not overwrite diagnostic selection')
    source_freeze = read(SOURCE / 'freeze.json')
    verify_hashes(source_freeze['dependencies_sha256'])
    repeat_freeze = read(RATINGS.parent / 'review-freeze.json')
    verify_hashes(repeat_freeze['sha256'])
    replies = lines(SOURCE / 'replies.jsonl')
    summary = read(SOURCE / 'summary.json')
    if file_hash(SOURCE / 'replies.jsonl') != summary['replies_sha256'] or len(replies) != 135:
        raise ValueError('Changed original replies')
    config = read(SOURCE / 'config.json')
    if file_hash(config['checkpoint']) != config['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from actual fifth evaluation')
    cases = choose_cases(replies, read(RATINGS))
    save(OUT / 'selection.json', cases)
    sources = [SOURCE / n for n in ('config.json', 'freeze.json', 'summary.json', 'replies.jsonl')]
    sources += [RATINGS, RATINGS.parent / 'review-freeze.json', RATINGS.parent / 'reviews.json',
                OUT / 'selection.json', Path(config['checkpoint']), Path(__file__),
                Path('docs/fifth-generation-diagnostic-protocol-v1.md')]
    sources += [Path('src/my_project_1') / n for n in ('generation_diagnostic.py', 'generate.py',
                'model.py', 'attention.py', 'transformer.py', 'tokenization.py')]
    artifact = Path(config['model_config']['tokenizer_artifact'])
    sources += [artifact / n for n in ('config.json', 'freeze.json', 'tokenizer.json', 'length-statistics.json')]
    save(OUT / 'freeze.json', {'generation': config, 'case_count': len(cases),
         'source_stop_counts_all135': dict(Counter(r['stop_reason'] for r in replies)),
         'dependencies_sha256': {str(p): file_hash(p) for p in sources},
         'torch_version': torch.__version__, 'tokenizers_version': tokenizers.__version__,
         'training_started': False, 'source': 'existing fifth best.pt, never v1.2 trained'})
    print('Frozen cases:', [(r['stratum'], r['id'].rsplit(':', 1)[-1]) for r in cases], flush=True)


def run():
    frozen = read(OUT / 'freeze.json')
    verify_hashes(frozen['dependencies_sha256'])
    config = frozen['generation']
    if frozen['torch_version'] != torch.__version__ or frozen['tokenizers_version'] != tokenizers.__version__:
        raise ValueError('Runtime changed after freeze')
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed'])
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
    model_config = checkpoint['model_config']
    if model_config != config['model_config']:
        raise ValueError('Wrong checkpoint model config')
    tokenizer, tc = load_frozen(model_config['tokenizer_artifact'])
    if tc['tokenizer_sha256'] != model_config['tokenizer_revision'] or tc['vocab_size'] != model_config['vocab_size']:
        raise ValueError('Wrong frozen tokenizer')
    model = TinyDecoderLM.from_config(model_config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    cases = read(OUT / 'selection.json')
    previous = {r['id']: r for r in lines(SOURCE / 'replies.jsonl')}
    reviews = {r['id']: r for r in read(RATINGS.parent / 'reviews.json') if r['model'] == 'fifth'}
    results = []
    for i, case in enumerate(cases, 1):
        path = OUT / 'traces' / f"{case['id'].rsplit(':', 1)[-1]}.json"
        if path.exists():
            item = read(path)
        else:
            result = trace_reply(model, tokenizer, model_config, case['instruction'],
                                 max_new_tokens=config['max_new_tokens'])
            old = previous[case['id']]
            exact = (result['response'] == old['generated_response'] and
                     result['generated_token_ids'] == old['generated_token_ids'] and
                     all(result[k] == old[k] for k in ('prompt_tokens', 'generated_tokens', 'stop_reason')))
            if not exact:
                save(OUT / 'failures' / path.name, {'case': case, 'result': result, 'old': old})
                raise ValueError('Replay differs from frozen fifth output')
            evidence = reviews[case['id']]['evidence']
            if case['stratum'] == 'refund-guarantee':
                evidence = ['ensure a smooth and satisfactory resolution'] if case['id'].endswith(':16355') else ['ensure a smooth resolution']
            spans = [locate_quote(tokenizer, result['generated_token_ids'], q) for q in evidence]
            item = {**case, **result, 'matches_original_exactly': exact, 'prior_review_evidence': spans}
            save(path, item)
        old = previous[case['id']]
        if item['instruction'] != case['instruction'] or item['generated_token_ids'] != old['generated_token_ids']:
            raise ValueError('Invalid resumed trace')
        if len(item['steps']) != item['generated_tokens']:
            raise ValueError('Incomplete trace')
        results.append({'id': item['id'], 'stratum': item['stratum'], 'stop_reason': item['stop_reason'],
                        'generated_tokens': item['generated_tokens'], 'context_remaining': item['context_remaining'],
                        'last_eos_probability': item['steps'][-1]['eos_probability'],
                        'last_eos_rank': item['steps'][-1]['eos_rank_strict'], 'path': str(path)})
        print(f"{i}/{len(cases)} {case['id'].rsplit(':', 1)[-1]} exact replay; {item['stop_reason']}; {item['generated_tokens']} tokens", flush=True)
    verify_hashes(frozen['dependencies_sha256'])
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Unexpected gradients')
    save(OUT / 'summary.json', {'cases': results, 'exact_replays': len(results),
         'source_stop_counts_all135': frozen['source_stop_counts_all135'],
         'stop_counts_cases': dict(Counter(r['stop_reason'] for r in results)),
         'training_started': False, 'optimizer_updates': 0, 'model_gradients_none': True,
         'trace_sha256': {r['path']: file_hash(r['path']) for r in results},
         'freeze_sha256': file_hash(OUT / 'freeze.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run'))
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else run)()
