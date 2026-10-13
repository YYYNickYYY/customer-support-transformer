"""Read-only acceptance of the frozen local candidate-pool audit."""
import json
from collections import Counter

import pytest

from my_project_1.beam_pool_audit import OUT, verify
from my_project_1.expanded_dev import read
from my_project_1.tokenization import file_hash


@pytest.fixture
def artifacts():
    if not (OUT / 'review.json').exists():
        pytest.skip('Local frozen audit artifacts are not distributed with source')
    return (verify(), [json.loads(s) for s in (OUT / 'candidates.jsonl').read_text(encoding='utf-8').splitlines()],
            read(OUT / 'review.json'))


def test_coverage_sources_and_no_checker(artifacts):
    plan, rows, review = artifacts
    assert len(rows) == 12 and len(review['ratings']) == 60
    assert Counter(r['intent'] for r in rows) == Counter({s['intent']: 2 for s in plan['samples']})
    for path, digest in review['sources_sha256'].items():
        assert file_hash(path) == digest
    summary = read(OUT / 'generation-summary.json')
    assert summary['weights_unchanged'] and summary['optimizer_updates'] == 0
    assert summary['candidates_sha256'] == file_hash(OUT / 'candidates.jsonl')
    for row, sample in zip(rows, plan['samples']):
        assert all(row[k] == sample[k] for k in ('id', 'instruction', 'input_ids'))
        assert row['checker_selection']['selected_rank'] is None
        assert row['checker_selection']['effectiveness'] is None
        assert len(row['beam']['candidates']) == 4


def test_scores_evidence_and_decision(artifacts):
    _, rows, review = artifacts
    keys = {(r['case'], r['candidate']) for r in review['ratings']}
    assert keys == {(i, k) for i in range(1, 13) for k in ('greedy', 'beam1', 'beam2', 'beam3', 'beam4')}
    for rating in review['ratings']:
        row = rows[rating['case'] - 1]
        result = row['greedy'] if rating['candidate'] == 'greedy' else row['beam']['candidates'][int(rating['candidate'][-1]) - 1]
        assert all(q and q in result['response'] for q in rating['evidence'])
        usable = (rating['intent_score'] == 2 and rating['helpfulness_score'] == 2
                  and rating['fluency_score'] >= 1 and not rating['unsupported_claim']
                  and not rating['entity_issue'] and not rating['severe_repetition'])
        assert rating['usable'] == usable
    assert review['top1_comparisons'] == {'better': 2, 'same_or_tradeoff': 8, 'worse': 2}
    assert review['metrics']['greedy']['usable'] == review['metrics']['beam1']['usable'] == 3
    assert sum(c['pool_usable'] for c in review['cases']) == review['pool_usable_questions'] == 3
    assert sum(c['reranking_opportunity'] for c in review['cases']) == review['reranking_opportunities'] == 0
    assert review['decision'] == 'stop_current_beam_checker_route_no_automatic_expansion'


def test_full_trace_reconstruction(artifacts):
    plan, rows, _ = artifacts
    alpha = plan['generation']['length_penalty']
    for row in rows:
        parents = {1: ([], 0.0)}
        for step in row['beam']['trace']:
            current = {}
            for node in step['kept']:
                tokens, logp = parents[node['parent_rank']]
                tokens = tokens if node['carried'] else tokens + [node['token']]
                logp = logp if node['carried'] else logp + node['token_log_probability']
                assert len(tokens) == node['length']
                assert logp == pytest.approx(node['log_probability'])
                assert node['score'] == pytest.approx(logp / len(tokens) ** alpha)
                current[node['rank']] = (tokens, logp)
            parents = current
        for candidate in row['beam']['candidates']:
            tokens, logp = parents[candidate['rank']]
            assert tokens == candidate['generated_token_ids']
            assert logp == pytest.approx(candidate['log_probability'])
            assert candidate['generated_tokens'] == len(tokens) <= plan['generation']['max_new_tokens']
            assert candidate['stop_reason'] in ('eos', 'max_new_tokens', 'context_limit')
