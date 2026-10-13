"""Read-only lexical provenance check; no inference, scores or training writes."""
import json
from pathlib import Path
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash, read_rows

BASE = Path('data/derived/refund-execution-v3.1/train.jsonl')
PREVIEW = Path('outputs/refund-supplement-20261006-v1.2')
REPLIES = Path('outputs/dev-expanded-fifth-20261005/replies.jsonl')
OUT = Path('outputs/refund-boundary-review-20261006/diagnostic-source-scan.json')
PHRASES = (
    'ensure a smooth and satisfactory resolution',
    'ensure a smooth and efficient resolution',
    'Your insights will help us expedite the process and ensure a smooth resolution.',
)


def main():
    if OUT.exists():
        raise ValueError('Do not overwrite diagnostic evidence')
    frozen = read(PREVIEW / 'manifest.json')
    for group in ('sha256', 'parent_dependency_sha256'):
        for p, h in frozen[group].items():
            if file_hash(p) != h:
                raise ValueError('Changed frozen input ' + p)
    changes = {r['id']: r for r in read(PREVIEW / 'changes-preview.json')}
    rows = list(read_rows(BASE))
    counts = []
    for phrase in PHRASES:
        before = [r for r in rows if phrase in r['response']]
        after = [r for r in rows if phrase in changes.get(r['id'], {}).get('after_response', r['response'])]
        counts.append({'phrase': phrase, 'actual_fifth_train_matching_records': len(before),
                       'untrained_v12_preview_matching_records': len(after),
                       'preview_remaining': [{'id': r['id'], 'intent': r['intent'],
                                              'response': r['response']} for r in after]})
    ids = {'16355', '26320', '26438'}
    # Generated reply JSONL has generated_response, not the training response contract.
    replies = [json.loads(line) for line in REPLIES.read_text(encoding='utf-8').splitlines()]
    examples = [r for r in replies if r['id'].rsplit(':', 1)[-1] in ids]
    assert len(examples) == 3
    paths = [BASE, PREVIEW / 'manifest.json', PREVIEW / 'changes-preview.json', REPLIES, Path(__file__)]
    save(OUT, {'method': 'Case-sensitive exact substring counts by record over all train',
               'counts': counts, 'existing_fifth_generated_examples': examples,
               'new_generation_performed': False, 'training_started': False,
               'preview_used_for_training': False,
               'interpretation_limit': 'Zero exact matches does not prove no semantic source or pure model cause. '
                                       'Remaining cross-intent phrases may transfer, but do not establish causality. '
                                       'Preview is untrained, not a model result.',
               'sha256': {str(p): file_hash(p) for p in paths}})
    print([(r['actual_fifth_train_matching_records'], r['untrained_v12_preview_matching_records'])
           for r in counts])


if __name__ == '__main__':
    main()
