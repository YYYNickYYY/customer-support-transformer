"""Validate explicit AI decisions and compare immutable same-ID dev outputs.

This only aggregates saved full-text judgments; it does not infer scores.
"""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash

OUT = Path('outputs/dev-expanded-fifth-20261005')
OLD = Path('outputs/dev-expanded-20261005-v1')
KEYS = ('intent_score', 'helpfulness_score', 'fluency_score', 'repetition', 'unsupported_claim')
MODELS = ('second', 'third', 'fourth', 'fifth')
FOCUS = ('check_refund_policy', 'get_refund', 'track_refund')

def validate_decision(decision, material):
    key, scores, evidence, notes = decision
    if key != material['review_id'] or len(scores) != 5:
        raise ValueError('Invalid review key or score count')
    if any(type(v) is not int or not 0 <= v <= (2 if i < 3 else 1) for i, v in enumerate(scores)):
        raise ValueError('Invalid score')
    if not evidence or not notes or any(not e or e not in material['generated_response'] for e in evidence):
        raise ValueError(f'{key}: evidence not in full response')
    return dict(zip(KEYS, scores))

def metrics(rows):
    if not rows:
        raise ValueError('No ratings')
    return {'n': len(rows), **{k: (sum(r[k] for r in rows) / len(rows) if i < 3 else sum(r[k] for r in rows)) for i, k in enumerate(KEYS)}}

def paired_counts(pairs):
    return {k: dict(zip(('improved', 'unchanged', 'worsened'),
        (sum((r[k] > 0 if i < 3 else r[k] < 0) for r in pairs),
         sum(r[k] == 0 for r in pairs),
         sum((r[k] < 0 if i < 3 else r[k] > 0) for r in pairs)))) for i, k in enumerate(KEYS)}

def main():
    for p, digest in read(OUT/'freeze.json')['dependencies_sha256'].items():
        assert file_hash(p) == digest, f'Changed generation dependency: {p}'
    for p, digest in read(OUT/'review-freeze.json')['sources_sha256'].items():
        assert file_hash(p) == digest, f'Changed review source: {p}'
    historical = read(OLD/'ai-expert-ratings.json')
    material = {r['review_id']: r for r in read(OUT/'review-material.json')}
    mapping = {r['review_id']: r for r in read(OUT/'review-mapping.json')}
    decisions_path = Path('.tools/dev-sheet-builder/fifth-decisions.json')
    focal_path = Path('.tools/dev-sheet-builder/fifth-refund-decisions.json')
    decisions = read(decisions_path)
    assert len(decisions) == 135 and len({r[0] for r in decisions}) == 135
    assert set(material) == set(mapping) == {r[0] for r in decisions}
    new = []
    for d in decisions:
        key = d[0]
        new.append({**mapping[key], **validate_decision(d, material[key]), 'model': 'fifth',
            'notes': d[3], 'evidence': d[2], 'reviewer': 'AI full-text review; risk v1.1 / other v1.0; not blind',
            'review_date': datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'risk_rubric': 'customer-support-risk-review-v1.1', 'other_rubric': 'customer-support-review-v1.0'})
    ratings = historical['ratings'] + new
    assert len(ratings) == 540 and len({(r['model'], r['id']) for r in ratings}) == 540
    raw = {m: [json.loads(l) for l in ((OUT if m == 'fifth' else OLD/m)/'replies.jsonl').read_text(encoding='utf-8').splitlines()] for m in MODELS}
    selection = read('artifacts/dev-check-expanded-v1/selection.json')['samples']
    for m in MODELS:
        assert [r['id'] for r in raw[m]] == [r['id'] for r in selection]
        for a, b in zip(raw['fifth'], raw[m]):
            assert all(a[k] == b[k] for k in ('id', 'instruction', 'reference_response', 'intent', 'cohort'))
    by_key = {(r['model'], r['id']): r for r in ratings}
    comparisons = list(historical['pairs'])
    for s in selection:
        a, b = by_key['fourth', s['id']], by_key['fifth', s['id']]
        comparisons.append({'from': 'fourth', 'to': 'fifth', 'cohort': s['cohort'], 'intent': s['intent'], 'id': s['id'], **{k: b[k]-a[k] for k in KEYS}})
    all_metrics = {m: {c: metrics([r for r in ratings if r['model'] == m and (c == 'all135' or r['cohort'] == c)]) for c in ('original27', 'new108', 'all135')} for m in MODELS}
    assert all(all_metrics[m] == historical['metrics'][m] for m in MODELS[:-1])
    per_intent = {m: {intent: metrics([by_key[m, s['id']] for s in selection if s['intent'] == intent]) for intent in sorted({s['intent'] for s in selection})} for m in MODELS}
    counts = dict(historical['paired_counts'])
    counts['fourth→fifth'] = {c: paired_counts([r for r in comparisons if r['from'] == 'fourth' and (c == 'all135' or r['cohort'] == c)]) for c in ('original27', 'new108', 'all135')}
    focal = []
    for m, decisions in read(focal_path).items():
        focus_raw = {int(r['id'].split(':')[-1]): r for r in raw[m] if r['intent'] in FOCUS}
        assert len(decisions) == len(focus_raw) == 15 and {d[0] for d in decisions} == set(focus_raw)
        for num, mark, quote, note in decisions:
            r = focus_raw[num]
            assert type(mark) is int and mark in (0, 1) and quote in r['generated_response'] and note
            focal.append({'model': m, 'id': r['id'], 'intent': r['intent'], 'cohort': r['cohort'],
                'refund_result_tag': mark, 'quote': quote, 'notes': note,
                'general_risk_unchanged': by_key[m, r['id']]['unsupported_claim']})
    focal_metrics = {m: {i: {'n': len(sub), 'refund_result_tag': sum(r['refund_result_tag'] for r in sub),
        **metrics([by_key[m, r['id']] for r in sub])} for i in (*FOCUS, 'all15')
        for sub in [[r for r in focal if r['model'] == m and (i == 'all15' or r['intent'] == i)]]} for m in MODELS}
    sources = [decisions_path, focal_path, OUT/'review-freeze.json', OUT/'calibration-review.json', OLD/'ai-expert-ratings.json',
        Path('docs/evaluation-risk-rubric-v1.1.md'), Path('docs/evaluation-rubric-v1.md'), Path(__file__)]
    payload = {'method': 'Explicit full-text AI review of 135 shuffled fifth outputs; known model, not blind or independent human. Historical405 ratings preserved exactly. Risk v1.1, other v1.0; no total score; focal tag is not sixth score.',
        'ratings': ratings, 'metrics': all_metrics, 'per_intent': per_intent,
        'pairs': comparisons, 'paired_counts': counts, 'focused_refund_audit': focal,
        'focused_refund_metrics': focal_metrics, 'stop_counts': {m: dict(Counter(r['stop_reason'] for r in raw[m])) for m in MODELS},
        'best_dev_loss': {'second': 1.0957509105295578, 'third': 1.095477457490304, 'fourth': 1.102186702290749, 'fifth': read(OUT/'summary.json')['training_summary']['best_dev_loss']},
        'sources_sha256': {str(p): file_hash(p) for p in sources},
        'replies_sha256': {m: file_hash((OUT if m == 'fifth' else OLD/m)/'replies.jsonl') for m in MODELS}}
    save(OUT/'ai-expert-ratings.json', payload)
    save(OUT/'rating-freeze.json', {'ratings_sha256': file_hash(OUT/'ai-expert-ratings.json'), 'sources_sha256': payload['sources_sha256'],
        'generation_freeze_sha256': file_hash(OUT/'freeze.json'), 'historical_workbook_sha256': file_hash(OLD/'expanded-dev-comparison.xlsx')})
    print(json.dumps({k: payload[k] for k in ('metrics', 'focused_refund_metrics', 'stop_counts')}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
