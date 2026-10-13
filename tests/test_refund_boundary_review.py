"""Boundary corrections stay evidence-only and preserve frozen history."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('boundary_review', ROOT / 'scripts/review_refund_boundary.py')
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


def fixture():
    reviews = [{'review_id': 'E033', 'approved': True, 'before': 'Help you navigate to secure reimbursement.',
                'after': 'We can review your request.', 'reason': 'Old reason', 'members': []}]
    decisions = {'reviewed_candidate_ids': ['E033'], 'overrides': {
        'E033': {'old_approved': True, 'approved': False, 'quote': 'navigate to secure',
                 'reason': '协助目的，与E002同尺度'}}}
    return reviews, decisions


def test_withdraw_only_explicit_edit_without_mutating_parent():
    reviews, decisions = fixture()
    old = deepcopy(reviews)
    result = boundary.corrected(reviews, decisions)
    assert reviews == old
    assert not result[0]['approved']
    assert result[0]['after'] == reviews[0]['before']
    assert result[0]['parent_after'] == reviews[0]['after']


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'old_decision', 'quote', 'new_approval', 'no_reason'])
def test_correction_rejects_invalid_evidence(defect):
    reviews, decisions = fixture()
    item = decisions['overrides']['E033']
    if defect == 'missing':
        decisions['reviewed_candidate_ids'] = []
    elif defect == 'duplicate':
        decisions['reviewed_candidate_ids'].append('E033')
    elif defect == 'old_decision':
        item['old_approved'] = False
    elif defect == 'quote':
        item['quote'] = 'invented'
    elif defect == 'new_approval':
        item['approved'] = True
    else:
        item['reason'] = ''
    with pytest.raises(ValueError):
        boundary.corrected(reviews, decisions)


@pytest.mark.skipif(not (ROOT / boundary.OUT / 'manifest.json').exists(),
                    reason='Local frozen audit snapshot not distributed')
def test_frozen_v12_replays_and_all_fulltexts_reviewed_without_activation(monkeypatch):
    from my_project_1.tokenization import file_hash, read_rows
    monkeypatch.chdir(ROOT)
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    directory = ROOT / boundary.OUT
    frozen = read(directory / 'manifest.json')
    for group in ('sha256', 'parent_dependency_sha256'):
        for p, h in frozen[group].items():
            assert file_hash(ROOT / p) == h, p
    assert not frozen['activated'] and not frozen['training_started']
    assert not frozen['training_activation_approved'] and not frozen['training_revision_created']
    assert frozen['approved_rules'] == 170 and frozen['approved_occurrences'] == 171
    assert frozen['reviewed_fulltext_records'] == 158
    assert frozen['pending_fulltext_ids'] == []
    parent = read(ROOT / boundary.PARENT / 'reviews-final.json')
    actual = read(directory / 'reviews-final.json')
    changed = {b['review_id'] for a, b in zip(parent, actual) if a != b}
    assert changed == {'E005', 'E033'}
    lookup = {r['review_id']: r for r in actual}
    assert not lookup['E002']['approved'] and not lookup['E005']['approved'] and not lookup['E033']['approved']
    helper = boundary.helper()
    rows = list(read_rows(ROOT / helper.BASE))
    output, changes = helper.replay(rows, actual)
    assert changes == read(directory / 'changes-preview.json')
    judgments = read(directory / 'fulltext-reviews.json')
    assert len(judgments) == len({r['id'] for r in judgments}) == 158
    material = {r['id']: r for r in read(directory / 'fulltext-material.json')}
    for r in judgments:
        assert r['status'] == 'pass'
        assert all(q in material[r['id']]['response'] for q in r['evidence'])
        assert not r['new_help_function_loss'] and not r['new_clear_sentence_seam']
    assert len(output) == len(rows) == 21528
