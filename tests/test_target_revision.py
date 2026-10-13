import json
from pathlib import Path

import pytest

from my_project_1.target_revision import derive, freeze_revision, load_revision, revise_response


@pytest.mark.parametrize('text', [
    "I'm here to ensure a seamless experience.",
    "We are committed to ensuring your satisfaction.",
    "We will do our best to ensure a smooth resolution.",
    "Our goal is to ensure a satisfactory resolution.",
    "We'll do everything we can to ensure a swift resolution.",
    "I'll do everything I can to ensure a satisfactory resolution.",
    "I will do my utmost to ensure a satisfactory resolution.",
    "Our team will work diligently to ensure a fair resolution.",
    "Your cooperation in this matter will ensure a smooth resolution.",
    "We are here to help ensure a smooth resolution.",
    "Double-check the address to ensure accuracy.",
    "If eligible, you may receive a refund.",
    "We accept Visa and deliver to {{Delivery City}}.",
    "Your complaint will be handled with the utmost care.",
])
def test_normal_intent_and_other_factors_unchanged(text):
    assert revise_response(text)[0] == text
    assert revise_response(text)[1] == []


def test_result_and_privacy_spans_without_broken_conjunctions():
    text = "I'm here to ensure a smooth and satisfactory resolution. We value your privacy, and we will handle your personal information with the utmost care and protection."
    changed, edits, _ = revise_response(text)
    assert changed == "I'm here to work towards a resolution. We value your privacy, and we recommend sharing only the necessary information through an official support channel."
    assert len(edits) == 2
    assert all(text[e['start']:e['end']] == e['before'] for e in edits)
    assert revise_response(changed)[0] == changed


def test_unapproved_continuation_is_not_rewritten():
    text = 'We will handle your information with the utmost care and an unknown safeguard.'
    assert revise_response(text)[0] == text


def test_coordinated_privacy_predicate_preserves_grammar():
    text = 'Rest assured, we value your privacy and will handle your information with the utmost care.'
    assert revise_response(text)[0] == 'Rest assured, we value your privacy and recommend sharing only the necessary information through an official support channel.'


def test_cancel_not_success_and_placeholders_preserved():
    text = 'With {{Order Number}}, I will ensure that your subscription is canceled promptly.'
    after, edits, _ = revise_response(text)
    assert after == 'With {{Order Number}}, I will help you submit your subscription cancellation request for review.'
    assert len(edits) == 1
    paired = 'I will ensure that your subscription is promptly canceled, and you will no longer receive our newsletter.'
    assert revise_response(paired)[0] == 'I will help you submit your subscription cancellation request for review.'


def test_capitalization_boundaries_and_no_in_word_match():
    assert revise_response('Ensure a fair resolution.')[0] == 'Work towards a resolution.'
    assert revise_response('reensure a fair resolution.')[0] == 'reensure a fair resolution.'


def test_derive_only_changes_response_and_preserves_order():
    rows = [{'id': 'a', 'intent': 'x', 'instruction': 'ID 37777', 'response': 'We ensure a fair resolution.', 'extra': 42},
            {'id': 'b', 'intent': 'y', 'instruction': 'hi', 'response': 'Here to help.'}]
    revised, changes, skipped = derive(rows)
    assert [r['id'] for r in revised] == ['a', 'b']
    assert {k: v for k, v in revised[0].items() if k != 'response'} == {k: v for k, v in rows[0].items() if k != 'response'}
    assert revised[1] == rows[1]
    assert len(changes) == 1 and not skipped
    assert rows[0]['response'] == 'We ensure a fair resolution.'
    with pytest.raises(ValueError, match='Duplicate'):
        derive(rows + rows[:1])


def test_freeze_replay_and_tamper_fail_closed(tmp_path):
    from my_project_1.tokenization import read_rows
    directory = tmp_path / 'revision'
    manifest = freeze_revision(directory)
    rows = list(read_rows('data/splits/bitext-v1/train.jsonl'))
    derived, stamp = load_revision(directory / 'manifest.json', rows)
    assert len(derived) == len(rows) == 21528
    assert manifest['changed_samples'] > 0
    assert stamp['actual_train_sha256'] == manifest['files_sha256']['train.jsonl']
    with pytest.raises(ValueError, match='refuse overwrite'):
        freeze_revision(directory)
    target = directory / 'train.jsonl'
    target.write_bytes(target.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_revision(directory / 'manifest.json', rows)


def test_dev_cannot_load_target_revision():
    from my_project_1.dataset import CustomerSupportDataset
    with pytest.raises(ValueError, match='only allowed for train'):
        CustomerSupportDataset('dev', train_revision='not-a-file')


def test_third_config_changes_only_revision_selection():
    from my_project_1.training import FullTrainConfig
    baseline = json.loads(Path('configs/training.json').read_text())
    third = json.loads(Path('configs/training-third.json').read_text())
    assert third.pop('train_revision') == 'data/derived/result-privacy-v1/manifest.json'
    assert third == baseline
    assert FullTrainConfig(**baseline).train_revision is None

