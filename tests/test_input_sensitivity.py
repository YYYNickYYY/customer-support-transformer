import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sensitivity", ROOT / "scripts/diagnose_input_sensitivity.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def plan():
    return m.r.read(m.OUT / "plan.json")


def test_preregistered_count_no_training_and_original_decoding():
    p = plan()
    assert len(m.stimuli(p)) == 22
    assert not p["training"] and p["no_repeat_ngram_size"] == 0


def test_negation_is_the_only_content_difference():
    groups = plan()["groups"]
    for index in (2, 3):
        a, b = groups[index]["cases"]
        assert b["instruction"].replace("do not want", "want") == a["instruction"]
        assert b["negation_required"]


def test_entity_pairs_change_only_the_identifier():
    for index in (4, 5):
        a, b = plan()["groups"][index]["cases"]
        assert a["instruction"].replace(a["required_id"], b["required_id"]) == b["instruction"]
        assert "include the" in a["instruction"]


def test_duplicate_control_and_paraphrase_expected_actions():
    groups = plan()["groups"]
    assert groups[8]["cases"][0]["instruction"] == groups[8]["cases"][1]["instruction"]
    for index in (6, 7):
        assert len({c["expected_action"] for c in groups[index]["cases"]}) == 1


def test_missing_or_duplicate_stimulus_rejected():
    p = plan()
    p["groups"][0]["cases"][0]["id"] = p["groups"][0]["cases"][1]["id"]
    with pytest.raises(ValueError):
        m.stimuli(p)
