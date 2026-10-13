"""Validate saved full-text AI judgments and summarize paired fifth/sixth results."""
from collections import Counter
import json
from pathlib import Path

from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash
from my_project_1.training_sixth import load_plan

OUT = Path('outputs/dev-expanded-sixth-20261007')
KEYS = ('intent_score', 'helpfulness_score', 'fluency_score', 'repetition', 'unsupported_claim')


def metrics(rows):
    return {'n': len(rows), **{k: sum(r[k] for r in rows) / len(rows) if i < 3 else sum(r[k] for r in rows)
                              for i, k in enumerate(KEYS)}}


def main():
    plan = load_plan()
    for p, h in read(OUT / 'freeze.json')['dependencies_sha256'].items():
        assert file_hash(p) == h, p
    raw = [json.loads(x) for x in (OUT / 'replies.jsonl').read_text(encoding='utf-8').splitlines()]
    oldraw = [json.loads(x) for x in Path(plan['baseline_replies']).read_text(encoding='utf-8').splitlines()]
    baseline = {r['id']: r for r in read(plan['baseline_ratings'])['ratings'] if r['model'] == 'fifth'}
    decisions = read(OUT / 'decisions.json')
    assert len(raw) == len(oldraw) == len(baseline) == len(decisions) == 135
    assert {d[0] for d in decisions} == set(range(1, 136))
    lookup = {d[0]: d for d in decisions}
    ratings, pairs = [], []
    for i, (r, old) in enumerate(zip(raw, oldraw), 1):
        assert all(r[k] == old[k] for k in ('id', 'intent', 'instruction', 'cohort', 'reference_response'))
        _, scores, quotes, note, entity_issue, tags = lookup[i]
        assert len(scores) == 5 and all(type(v) is int and 0 <= v <= (2 if j < 3 else 1) for j, v in enumerate(scores))
        assert quotes and all(q and q in r['generated_response'] for q in quotes), i
        assert note and type(entity_issue) is int and entity_issue in (0, 1)
        scored = {k: r[k] for k in ('id', 'intent', 'cohort', 'stop_reason')}
        scored.update(dict(zip(KEYS, scores)), evidence=quotes, notes=note, entity_issue=entity_issue,
                      error_tags=tags, model='sixth', reviewer='AI full-text nonblind review; no keyword scoring',
                      review_date='2026-10-07', risk_rubric='customer-support-risk-review-v1.1',
                      other_rubric='customer-support-review-v1.0')
        ratings.append(scored)
        pairs.append({'id': r['id'], 'intent': r['intent'], 'cohort': r['cohort'],
                      **{k: scored[k] - baseline[r['id']][k] for k in KEYS}})
    counts = {k: dict(improved=sum((p[k] > 0 if j < 3 else p[k] < 0) for p in pairs),
                      unchanged=sum(p[k] == 0 for p in pairs),
                      worsened=sum((p[k] < 0 if j < 3 else p[k] > 0) for p in pairs)) for j, k in enumerate(KEYS)}
    all_metrics = {'fifth': metrics(list(baseline.values())), 'sixth': metrics(ratings)}
    sources = ['docs/sixth-learning-rate-protocol-v1.md', plan['baseline_ratings'], plan['baseline_replies'],
               str(OUT / 'replies.jsonl'), str(OUT / 'decisions.json'), str(OUT / 'freeze.json'), str(Path(__file__)),
               'docs/evaluation-risk-rubric-v1.1.md', 'docs/evaluation-rubric-v1.md']
    cohorts = {m: {c: metrics([r for r in rows if r['cohort'] == c]) for c in ('original27', 'new108')}
               for m, rows in [('fifth', list(baseline.values())), ('sixth', ratings)]}
    intents = {m: {intent: metrics([r for r in rows if r['intent'] == intent]) for intent in sorted({r['intent'] for r in rows})}
               for m, rows in [('fifth', list(baseline.values())), ('sixth', ratings)]}
    payload = {'method': 'Same fixed 135 dev IDs; sixth independently read before revealing paired scores; model identity known, not blind or independent human. Fifth frozen corrected scores unchanged. No total score.',
               'ratings': ratings, 'metrics': all_metrics, 'paired_counts': counts, 'pairs': pairs,
               'per_cohort': cohorts, 'per_intent': intents, 'entity_issues_sixth': sum(r['entity_issue'] for r in ratings),
               'stop_counts': {m: dict(Counter(r['stop_reason'] for r in rows)) for m, rows in [('fifth', oldraw), ('sixth', raw)]},
               'best_dev_loss': {'fifth': plan['baseline_best_dev_loss'], 'sixth': read(OUT / 'summary.json')['training_summary']['best_dev_loss']},
               'sources_sha256': {p: file_hash(p) for p in sources}}
    save(OUT / 'ai-expert-ratings.json', payload)
    save(OUT / 'rating-freeze.json', {'ratings_sha256': file_hash(OUT / 'ai-expert-ratings.json'),
                                    'sources_sha256': payload['sources_sha256'], 'complete': True})
    print(json.dumps({k: payload[k] for k in ('metrics', 'paired_counts', 'stop_counts', 'entity_issues_sixth')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
