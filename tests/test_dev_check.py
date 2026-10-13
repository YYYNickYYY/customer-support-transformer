import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from my_project_1.dev_check import RATING_FIELDS, choose_rows, csv_safe, frozen_selection, rating_record


def rows():
    return [{"id": str(i), "intent": "a" if i < 2 else "b", "category": "C",
             "template_id": str(i), "instruction": "question", "response": "answer"}
            for i in range(4)]


def test_selection_independent_of_order_and_response():
    original = rows()
    selected = choose_rows(original, 42)
    assert [row["intent"] for row in selected] == ["a", "b"]
    changed = [{**row, "response": "changed"} for row in reversed(original)]
    assert [r["id"] for r in choose_rows(changed, 42)] == [r["id"] for r in selected]


def test_selection_freeze_reuse_and_tamper_detection(tmp_path):
    dataset = SimpleNamespace(rows=rows(), config={"split_sha256": {"dev.jsonl": "fixedhash"}})
    directory = tmp_path / "selection"
    selected = frozen_selection(dataset, directory, 42, create=True)
    before = (directory / "selection.json").read_bytes()
    assert frozen_selection(dataset, directory, 42, create=True) == selected
    assert (directory / "selection.json").read_bytes() == before
    with pytest.raises(ValueError, match="does not match"):
        frozen_selection(dataset, directory, 43)
    (directory / "selection.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        frozen_selection(dataset, directory, 42)


def test_human_fields_blank_and_csv_formula_protection():
    record = rating_record(rows()[0], {"response": "output", "stop_reason": "eos", "generated_tokens": 4})
    assert all(record[field] == "" for field in RATING_FIELDS)
    assert csv_safe("=SUM(A1)") == "'=SUM(A1)"
    assert csv_safe("@evil") == "'@evil"
    assert csv_safe(0) == 0
    assert csv_safe("normal reply") == "normal reply"


def test_invalid_selection_input():
    with pytest.raises(ValueError, match="Empty"):
        choose_rows([])
    with pytest.raises(ValueError, match="Duplicate"):
        choose_rows([rows()[0], rows()[0]])


def test_frozen_actual_dev_coverage():
    from my_project_1.dataset import CustomerSupportDataset
    dataset = CustomerSupportDataset("dev")
    selected = frozen_selection(dataset, "artifacts/dev-check-v1", 42)
    assert len(selected) == len({row["intent"] for row in dataset.rows}) == 27
    assert len({row["id"] for row in selected}) == 27
    assert len({row["category"] for row in selected}) == 11
