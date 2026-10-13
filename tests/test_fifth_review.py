import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fifth_review', ROOT/'scripts/assemble_fifth_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

def example():
    return {'review_id': 'F001', 'generated_response': 'Please contact official support.'}

def test_explicit_scores_and_exact_evidence():
    d = ['F001', [2, 1, 2, 0, 0], ['official support'], '部分帮助']
    assert review.validate_decision(d, example())['helpfulness_score'] == 1
    d[2] = ['invented evidence']
    with pytest.raises(ValueError, match='evidence'):
        review.validate_decision(d, example())

@pytest.mark.parametrize('scores', [[2, 2, 2, 0, True], [2, 2, 2, 2, 0], [2, None, 2, 0, 0], [2, 2]])
def test_reject_missing_invalid_or_boolean_scores(scores):
    with pytest.raises(ValueError):
        review.validate_decision(['F001', scores, ['official'], 'note'], example())

def test_independent_metric_directions_and_pair_counts():
    rows = [dict(zip(review.KEYS, scores)) for scores in ([2, 0, 2, 0, 1], [1, 2, 1, 1, 0])]
    m = review.metrics(rows)
    assert m == {'n': 2, 'intent_score': 1.5, 'helpfulness_score': 1.0, 'fluency_score': 1.5, 'repetition': 1, 'unsupported_claim': 1}
    pairs = [dict(zip(review.KEYS, delta)) for delta in ([1, 0, -1, -1, 1], [-1, 1, 0, 0, -1])]
    c = review.paired_counts(pairs)
    assert c['repetition'] == {'improved': 1, 'unchanged': 1, 'worsened': 0}
    assert c['unsupported_claim'] == {'improved': 1, 'unchanged': 0, 'worsened': 1}

def test_empty_scores_not_zero():
    with pytest.raises(ValueError, match='No ratings'):
        review.metrics([])
