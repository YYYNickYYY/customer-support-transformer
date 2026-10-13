import json
from pathlib import Path

import pytest
from my_project_1.target_revision_v2 import derive, propose_sentence, sentence_spans, SEAMS, approved_rules


def test_passive_privacy_whole_sentence():
    after, kind, _ = propose_sentence('Your data will be handled with the utmost care.', 'create_account')
    assert kind == 'privacy-whole-sentence'
    assert after.endswith('channel.')
    assert 'will be handled' not in after


@pytest.mark.parametrize('text', [
    'We prioritize protecting your information.',
    'We are committed to ensuring your satisfaction.',
    "I'll do everything I can to ensure a fair resolution.",
    'Please create a secure password.',
    'We will work closely with you to ensure a resolution.',
])
def test_no_normal_service_proposal(text):
    assert propose_sentence(text, 'create_account')[0] is None


def test_numbered_step_is_not_split_from_text():
    spans = list(sentence_spans('1. Sign in.\n2. Open Orders.'))
    assert [s for _, _, s in spans] == ['1. Sign in.', '2. Open Orders.']


def test_seam_fixed_without_work_to_work():
    for before, after in SEAMS.items():
        assert propose_sentence(before, 'x')[0] == after
        assert 'work to work' not in after


def test_whole_sentence_ledger_preserves_other_text_and_fields():
    text = 'Hello. Your data will be handled with the utmost care. Keep {{Order Number}}.'
    rows = [{'id': '1', 'intent': 'create_account', 'instruction': 'hello', 'response': text}]
    rule = {'id': 'r', 'intent': '*', 'before': 'Your data will be handled with the utmost care.', 'after': 'Please use an official channel.', 'kind': 'privacy'}
    result, ledger = derive(rows, [rule])
    assert result[0]['response'] == 'Hello. Please use an official channel. Keep {{Order Number}}.'
    assert result[0]['instruction'] == 'hello'
    assert rows[0]['response'] == text
    for e in ledger[0]['edits']:
        assert text[e['start']:e['end']] == e['before']
    with pytest.raises(ValueError, match='Duplicate'):
        derive(rows, [rule, rule])
    with pytest.raises(ValueError, match='no match'):
        derive(rows, [{**rule, 'before': 'not present'}])


def test_reviewed_exclusions_and_required_seams():
    from my_project_1.target_revision_v2 import PROPOSALS
    proposal = json.loads((PROPOSALS / 'proposals.json').read_text(encoding='utf-8'))
    approval = json.loads((PROPOSALS / 'approval.json').read_text(encoding='utf-8'))
    rules = approved_rules(proposal, approval)
    assert set(SEAMS).issubset({r['before'] for r in rules})
    assert all('will handle your request' not in r['before'].lower() for r in rules if r['kind'] == 'privacy-whole-sentence')
    assert any('do my utmost' in r['before'] for r in approval['rejected'])


def test_fourth_only_changes_revision_and_has_task():
    third = json.loads(Path('configs/training-third.json').read_text())
    fourth = json.loads(Path('configs/training-fourth.json').read_text())
    assert fourth.pop('train_revision') == 'data/derived/result-privacy-v2/manifest.json'
    third.pop('train_revision')
    assert fourth == third
    tasks = json.loads(Path('.vscode/tasks.json').read_text(encoding='utf-8'))['tasks']
    task = next(t for t in tasks if t['label'].startswith('第四次训练'))
    assert task['args'][-1] == 'configs/training-fourth.json'
    assert task['dependsOn'] == '核验第四次冻结清单（不训练）'


def test_frozen_v2_replay_and_tamper_detection(tmp_path):
    import shutil
    from my_project_1.target_revision_v2 import OUT, load_revision, freeze_revision
    from my_project_1.tokenization import read_rows, file_hash
    source = list(read_rows('data/splits/bitext-v1/train.jsonl'))
    derived, provenance = load_revision(OUT / 'manifest.json', source)
    assert len(derived) == 21528
    assert provenance['actual_train_sha256'] == file_hash(OUT / 'train.jsonl')
    with pytest.raises(ValueError, match='refuse overwrite'):
        freeze_revision()
    destination = tmp_path / 'copy'
    shutil.copytree(OUT, destination)
    changes = destination / 'changes.jsonl'
    changes.write_bytes(changes.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='artifact mismatch'):
        load_revision(destination / 'manifest.json', source)
