"""Guard evidence-only review and replay; these tests never activate training."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repeat = load_script('finalize_repeat_review')
supplement = load_script('finalize_refund_supplement')


def review_fixture():
    material, mapping, decisions, ratings = [], [], [], []
    for model in ('fourth', 'fifth'):
        for i in range(135):
            rid = f'P{len(material) + 1:03}'
            material.append({'review_id': rid, 'response': 'Please check your request.'})
            mapping.append({'review_id': rid, 'model': model, 'id': str(i),
                            'cohort': 'original27' if i < 27 else 'new108', 'intent': 'get_refund'})
            decisions.append([rid, int(model == 'fifth'), ['check'], '有原文证据'])
            ratings.append({'model': model, 'id': str(i), 'repetition': 0,
                            'helpfulness_score': 2, 'unsupported_claim': 1})
    return material, mapping, decisions, {'ratings': ratings, 'best_dev_loss': {'fourth': 1, 'fifth': 2}}


def test_repetition_complete_pairing_and_preserves_other_fields():
    material, mapping, decisions, old = review_fixture()
    original = deepcopy(old)
    result, reviews = repeat.assemble(material, mapping, decisions, old)
    assert old == original
    assert len(reviews) == 270
    assert result['paired_repetition'] == {'0->1': 135}
    assert all(r['helpfulness_score'] == 2 and r['unsupported_claim'] == 1 for r in result['ratings'])


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'invented_quote', 'boolean_score'])
def test_repetition_rejects_invalid_review(defect):
    material, mapping, decisions, old = review_fixture()
    if defect == 'missing':
        decisions.pop()
    elif defect == 'duplicate':
        decisions[-1] = decisions[0]
    elif defect == 'invented_quote':
        decisions[0][2] = ['not present']
    else:
        decisions[0][1] = True
    with pytest.raises(AssertionError):
        repeat.assemble(material, mapping, decisions, old)


def replay_fixture():
    before = 'We guarantee your refund.'
    rows = [{'id': 'A', 'intent': 'get_refund', 'instruction': 'Refund please',
             'response': before + ' Keep {{Order Number}}.'}]
    reviews = [{'approved': True, 'review_id': 'S001', 'before': before,
                'after': 'We can help check your request.', 'reason': '核查表述',
                'members': [{'id': 'A', 'start': 0, 'end': len(before)}]}]
    return rows, reviews


def test_replay_preserves_original_entities_and_metadata():
    rows, reviews = replay_fixture()
    original = deepcopy(rows)
    output, changes = supplement.replay(rows, reviews)
    assert rows == original
    assert output[0]['response'] == 'We can help check your request. Keep {{Order Number}}.'
    assert {k: v for k, v in output[0].items() if k != 'response'} == {
        k: v for k, v in rows[0].items() if k != 'response'}
    assert len(changes) == 1


@pytest.mark.parametrize('defect', ['unreviewed_occurrence', 'missing_occurrence', 'entity', 'duplicate'])
def test_replay_rejects_unreviewed_or_damaging_change(defect):
    rows, reviews = replay_fixture()
    if defect == 'unreviewed_occurrence':
        rows.append({**rows[0], 'id': 'B'})
    elif defect == 'missing_occurrence':
        reviews[0]['members'][0]['start'] = 1
    elif defect == 'entity':
        reviews[0]['after'] = 'We can help {{Customer Name}}.'
    else:
        rows[0]['response'] += ' ' + reviews[0]['after']
    with pytest.raises(ValueError):
        supplement.replay(rows, reviews)


@pytest.mark.skipif(not (ROOT / 'outputs/refund-supplement-20261006-v1.1/manifest.json').exists(),
                    reason='Local frozen audit artifacts are not distributed with source code')
def test_local_frozen_outputs_match_hashes_and_replay():
    from my_project_1.tokenization import file_hash, read_rows
    def read(p):
        return json.loads(p.read_text(encoding='utf-8'))
    for folder, filename in [('refund-supplement-20261006-v1.1', 'manifest.json'),
                             ('repetition-review-20261006-v1', 'review-freeze.json')]:
        frozen = read(ROOT / 'outputs' / folder / filename)
        for group in ('sha256', 'dependency_sha256'):
            for p, h in frozen.get(group, {}).items():
                assert file_hash(ROOT / p) == h, p
    directory = ROOT / 'outputs/refund-supplement-20261006-v1.1'
    frozen = read(directory / 'manifest.json')
    assert not frozen['activated'] and not frozen['training_started']
    rows = list(read_rows(ROOT / supplement.BASE))
    output, changes = supplement.replay(rows, read(directory / 'reviews-final.json'))
    assert changes == read(directory / 'changes-preview.json')
    assert len(output) == 21528 and len(changes) == 158
