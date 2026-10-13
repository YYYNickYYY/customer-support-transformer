"""Checks for diagnostic helpers, not an automatic language-quality judge."""
import importlib.util
from pathlib import Path


def helpers():
    spec = importlib.util.spec_from_file_location('target_audit_helpers', Path('scripts/audit_target_revision.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_context_includes_immediate_neighbours():
    text = 'We value privacy. Your information will be protected. Thank you.'
    result = helpers().context_at(text, text.index('information'))
    assert result == {'previous': 'We value privacy.', 'sentence': 'Your information will be protected.', 'next': 'Thank you.'}


def test_candidate_flags_are_not_risk_scores():
    signals = helpers().signals
    assert signals('Your data will be handled securely.') == ['privacy']
    assert signals('We ensure your subscription is cancelled.') == ['result']
    assert signals('We aim to ensure a seamless resolution.') == ['result']  # deliberate false positive
    assert signals('Please choose a secure password and enter your details.') == ['privacy']  # not a guarantee
    assert signals('How can I help?') == []


def test_id_ranking_is_stable_and_content_independent():
    digest = helpers().digest
    rows = [{'id': 'b', 'response': 'guarantee'}, {'id': 'a', 'response': 'no guarantee'}]
    rank = lambda r: (digest(('target-revision-audit-v1:42:' + r['id']).encode()), r['id'])
    ids = [r['id'] for r in sorted(rows, key=rank)]
    rows.reverse()
    for r in rows:
        r['response'] = 'completely changed'
    assert [r['id'] for r in sorted(rows, key=rank)] == ids
