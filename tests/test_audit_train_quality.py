"""Checks for the audit's lexical candidate extraction, not automatic risk grading."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_train_quality.py'
spec = importlib.util.spec_from_file_location('audit_train_quality', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_goal_can_be_candidate_without_becoming_risk_score():
    found = audit.signals('Our goal is to ensure a seamless experience.')
    assert found['ensure_smooth_or_satisfaction']
    assert found['uncertainty_or_goal']
    assert 'unsupported_claim' not in found


def test_ensure_and_experience_must_be_same_sentence():
    assert not audit.signals('Please ensure the address is correct. A smooth experience is our goal.')['ensure_smooth_or_satisfaction']


def test_conditional_refund_is_still_only_candidate():
    found = audit.signals('If eligible, we may process a refund.')
    assert found['refund_action'] and found['uncertainty_or_goal']


def test_service_intent_has_no_lexical_guarantee():
    assert not audit.signals("I'm here to help you.")['guarantee_or_promise']


def test_id_order_is_content_independent():
    rows = [{'id': 'a', 'response': 'guaranteed refund'}, {'id': 'b', 'response': 'hello'}]
    ordered = lambda items: sorted(items, key=lambda r: (audit.digest(('train-quality-audit-v1:42:' + r['id']).encode()), r['id']))
    before = [r['id'] for r in ordered(rows)]
    rows[0]['response'], rows[1]['response'] = rows[1]['response'], rows[0]['response']
    assert [r['id'] for r in ordered(rows)] == before
