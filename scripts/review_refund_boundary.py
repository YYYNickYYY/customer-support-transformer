"""Create evidence-only v1.2 preview; never activates or trains a model."""
import argparse
import importlib.util
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash, read_rows, load_frozen, prepare_example

PARENT = Path('outputs/refund-supplement-20261006-v1.1')
OUT = Path('outputs/refund-supplement-20261006-v1.2')
PROTOCOL = Path('docs/refund-boundary-review-protocol-20261006.md')


def helper():
    spec = importlib.util.spec_from_file_location('refund_replay', 'scripts/finalize_refund_supplement.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_parent():
    manifest = read(PARENT / 'manifest.json')
    for group in ('sha256', 'dependency_sha256'):
        for p, h in manifest.get(group, {}).items():
            if file_hash(p) != h:
                raise ValueError('Changed frozen parent: ' + p)
    return manifest


def corrected(reviews, decisions):
    result = deepcopy(reviews)
    if {r['review_id'] for r in reviews} != set(decisions['reviewed_candidate_ids']):
        raise ValueError('All candidate sentence decisions must be covered')
    if len(decisions['reviewed_candidate_ids']) != len(reviews):
        raise ValueError('Duplicate reviewed ID')
    lookup = {r['review_id']: r for r in result}
    for rid, d in decisions['overrides'].items():
        row = lookup[rid]
        if d['old_approved'] != row['approved'] or d['quote'] not in row['before']:
            raise ValueError('Incorrect source decision/evidence')
        if d['approved'] is not False or not d['reason']:
            raise ValueError('This review only withdraws explicitly misclassified edits')
        row['parent_approved'] = row['approved']
        row['parent_after'] = row['after']
        row['approved'] = False
        row['after'] = row['before']
        row['reason'] = d['reason']
        row['boundary_review_quote'] = d['quote']
    return result


def prepare():
    if OUT.exists():
        raise ValueError('Do not overwrite an audit snapshot')
    parent = verify_parent()
    h = helper()
    h.verify_material()
    decisions = read('outputs/refund-boundary-review-20261006/decisions.json')
    reviews = corrected(read(PARENT / 'reviews-final.json'), decisions)
    rows = list(read_rows(h.BASE))
    output, changes = h.replay(rows, reviews)
    for name, data in [('reviews-final.json', reviews),
                       ('approved-rules.json', [r for r in reviews if r['approved']]),
                       ('changes-preview.json', changes)]:
        save(OUT / name, data)
    material = [{'id': c['id'], 'intent': c['intent'], 'instruction': c['instruction'],
                 'before_response': c['before_response'], 'response': c['after_response'],
                 'edits': c['edits']} for c in changes]
    save(OUT / 'fulltext-material.json', material)
    tok, cfg = load_frozen('artifacts/tokenizer/bitext-bpe-v1')
    changed = {r['id'] for r in changes}
    checks = []
    for b, a in zip(rows, output):
        if b['id'] not in changed:
            continue
        old = prepare_example(tok, b['instruction'], b['response'], cfg['max_length'])
        new = prepare_example(tok, a['instruction'], a['response'], cfg['max_length'])
        if new['supervised_tokens'] < 1:
            raise ValueError('No response supervision')
        checks.append({'id': b['id'], 'before_supervised_tokens': old['supervised_tokens'],
                       'after_supervised_tokens': new['supervised_tokens'],
                       'before_truncated': old['truncated'], 'after_truncated': new['truncated']})
    save(OUT / 'token-check.json', checks)
    files = [OUT / n for n in ('reviews-final.json', 'approved-rules.json', 'changes-preview.json',
                               'fulltext-material.json', 'token-check.json')]
    files += [PARENT / 'manifest.json', PROTOCOL, Path(__file__),
              Path('outputs/refund-boundary-review-20261006/decisions.json')]
    verify_parent()
    save(OUT / 'material-freeze.json', {'version': 'refund-supplement-list-v1.2',
         'activated': False, 'training_started': False, 'fulltext_acceptance_complete': False,
         'approved_rules': sum(r['approved'] for r in reviews),
         'approved_occurrences': sum(len(c['edits']) for c in changes),
         'changed_samples_preview': len(changes),
         'changed_intents': dict(Counter(c['intent'] for c in changes)),
         'parent_dependency_sha256': {**parent['dependency_sha256'], **parent['sha256']},
         'sha256': {str(p): file_hash(p) for p in files}})
    print('Evidence-only preview:', len(changes), 'full responses; no training')


def finalize():
    frozen = read(OUT / 'material-freeze.json')
    for p, h in frozen['sha256'].items():
        if file_hash(p) != h:
            raise ValueError('Changed audit material ' + p)
    verify_parent()
    material = read(OUT / 'fulltext-material.json')
    decisions = [r for p in sorted(OUT.glob('fulltext-decisions-*.json')) for r in read(p)]
    expected = {r['id']: r for r in material}
    if len(decisions) != len(expected) or {r['id'] for r in decisions} != set(expected):
        raise ValueError('Full text review missing or duplicated')
    for r in decisions:
        if not r['reason'] or not r['evidence'] or any(q not in expected[r['id']]['response'] for q in r['evidence']):
            raise ValueError('Missing exact evidence')
        if r['status'] not in ('pass', 'followup'):
            raise ValueError('Unknown semantic decision')
    checks = read(OUT / 'token-check.json')
    save(OUT / 'fulltext-reviews.json', decisions)
    pending = [r['id'] for r in decisions if r['status'] == 'followup']
    files = [OUT / 'material-freeze.json', OUT / 'fulltext-reviews.json',
             *sorted(OUT.glob('fulltext-decisions-*.json'))]
    save(OUT / 'manifest.json', {**frozen, 'frozen': True, 'fulltext_acceptance_complete': True,
         'training_revision_created': False, 'training_activation_approved': False,
         'pending_fulltext_ids': pending, 'reviewed_fulltext_records': len(decisions),
         'truncated_before': sum(r['before_truncated'] for r in checks),
         'truncated_after': sum(r['after_truncated'] for r in checks),
         'semantic_limit': 'AI scoped intervention review, not independent safety certification',
         'sha256': {**frozen['sha256'], **{str(p): file_hash(p) for p in files}}})
    print('Fulltext reviewed:', len(decisions), '; followup:', len(pending), '; no training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'finalize'))
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else finalize)()
