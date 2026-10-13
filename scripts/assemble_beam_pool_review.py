"""Aggregate explicit AI full-text decisions; no semantic scoring or checker."""
from collections import Counter
import json
from pathlib import Path

from my_project_1.beam_pool_audit import OUT, verify
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash

KEYS = ('intent_score', 'helpfulness_score', 'fluency_score', 'repetition', 'unsupported_claim')


def usable(r):
    return (r['intent_score'] == 2 and r['helpfulness_score'] == 2 and r['fluency_score'] >= 1
            and r['unsupported_claim'] == 0 and r['entity_issue'] == 0 and not r['severe_repetition'])


def compare(new, old):
    main = ('intent_score', 'helpfulness_score')
    guards = ('unsupported_claim', 'entity_issue', 'severe_repetition')
    if (all(new[k] >= old[k] for k in main) and any(new[k] > old[k] for k in main)
            and all(new[k] <= old[k] for k in guards) and new['fluency_score'] >= old['fluency_score']):
        return 'better'
    if any(new[k] < old[k] for k in main):
        return 'worse'
    return 'same_or_tradeoff'


def metrics(rows):
    return {'n': len(rows), **{k: sum(r[k] for r in rows) / len(rows) if i < 3 else sum(r[k] for r in rows)
                             for i, k in enumerate(KEYS)}, 'entity_issues': sum(r['entity_issue'] for r in rows),
            'severe_repetitions': sum(r['severe_repetition'] for r in rows), 'usable': sum(usable(r) for r in rows)}


def main():
    plan = verify()
    raw_path = OUT / 'candidates.jsonl'
    rows = [json.loads(x) for x in raw_path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 12
    assert file_hash(raw_path) == read(OUT / 'generation-summary.json')['candidates_sha256']
    decisions = read(OUT / 'decisions.json')
    expected = {(i, label) for i in range(1, 13) for label in ('greedy', 'beam1', 'beam2', 'beam3', 'beam4')}
    assert len(decisions) == 60 and {(d[0], d[1]) for d in decisions} == expected
    ratings = []
    for i, label, scores, entity, severe, quotes, note in decisions:
        raw = rows[i - 1]
        assert raw['id'] == plan['samples'][i - 1]['id']
        result = raw['greedy'] if label == 'greedy' else raw['beam']['candidates'][int(label[-1]) - 1]
        text = result['response']
        assert len(scores) == 5 and all(type(v) is int and 0 <= v <= (2 if j < 3 else 1) for j, v in enumerate(scores))
        assert type(entity) is int and entity in (0, 1) and type(severe) is int and severe in (0, 1)
        assert quotes and all(q and q in text for q in quotes) and note, (i, label)
        r = {'case': i, 'id': raw['id'], 'intent': raw['intent'], 'candidate': label,
             **dict(zip(KEYS, scores)), 'entity_issue': entity, 'severe_repetition': severe,
             'evidence': quotes, 'notes': note, 'stop_reason': result['stop_reason'],
             'risk_rubric': 'customer-support-risk-review-v1.1', 'other_rubric': 'customer-support-review-v1.0'}
        r['usable'] = usable(r)
        ratings.append(r)
    by_key = {(r['case'], r['candidate']): r for r in ratings}
    cases = []
    for i, raw in enumerate(rows, 1):
        old, top = by_key[i, 'greedy'], by_key[i, 'beam1']
        ranks = [j for j in range(1, 5) if usable(by_key[i, 'beam' + str(j)])]
        cases.append({'case': i, 'id': raw['id'], 'intent': raw['intent'],
                      'top1_vs_greedy': compare(top, old), 'greedy_usable': usable(old), 'top1_usable': usable(top),
                      'usable_beam_ranks': ranks, 'pool_usable': bool(ranks),
                      'reranking_opportunity': not usable(top) and bool(ranks),
                      'all_beams_wrong_intent': all(by_key[i, 'beam' + str(j)]['intent_score'] < 2 for j in range(1, 5)),
                      'unique_texts': raw['beam']['unique_response_count']})
    pool_count = sum(c['pool_usable'] for c in cases)
    opportunities = sum(c['reranking_opportunity'] for c in cases)
    recommend = pool_count >= 6 and opportunities >= 3
    payload = {'version': 'beam-pool-audit-v1', 'reviewer': 'AI full-text nonblind, known project/model history',
        'ratings': ratings, 'cases': cases, 'top1_comparisons': dict(Counter(c['top1_vs_greedy'] for c in cases)),
        'metrics': {label: metrics([r for r in ratings if r['candidate'] == label]) for label in ('greedy', 'beam1', 'beam2', 'beam3', 'beam4')},
        'pool_usable_questions': pool_count, 'reranking_opportunities': opportunities,
        'checker_selection': {'status': 'not_implemented', 'effectiveness': None},
        'decision': 'recommend_separate_checker_design' if recommend else 'stop_current_beam_checker_route_no_automatic_expansion',
        'sources_sha256': {str(p): file_hash(p) for p in [raw_path, OUT / 'decisions.json', OUT / 'freeze.json',
                            Path(__file__), Path('docs/evaluation-rubric-v1.md'), Path('docs/evaluation-risk-rubric-v1.1.md')]}}
    save(OUT / 'review.json', payload)
    print(json.dumps({k: payload[k] for k in ('top1_comparisons', 'metrics', 'pool_usable_questions', 'reranking_opportunities', 'decision')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
