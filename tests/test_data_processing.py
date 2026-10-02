import pytest

from my_project_1.data_processing import (
    normalize_tags,
    normalize_text,
    stable_ticket_id,
    transform_bitext_record,
    transform_bitext_records,
)


@pytest.fixture
def bitext_row():
    return {
        "flags": "refund, urgent | refund",
        "instruction": "  I   need\n a refund.  ",
        "category": "PAYMENT",
        "intent": "request_refund",
        "response": " We can help. ",
    }


def test_normalize_text_collapses_whitespace_and_normalizes_unicode():
    assert normalize_text("  Ａ\n  B ") == "A B"


def test_normalize_tags_splits_deduplicates_and_sorts():
    assert normalize_tags("refund, urgent | refund") == ["refund", "urgent"]


def test_stable_ticket_id_is_deterministic(bitext_row):
    assert stable_ticket_id(bitext_row) == stable_ticket_id(dict(bitext_row))
    assert stable_ticket_id(bitext_row).startswith("bitext-")


def test_transform_bitext_record_maps_to_work_order(bitext_row):
    ticket = transform_bitext_record(bitext_row, row_number=7)

    assert ticket["customer_message"] == "I need a refund."
    assert ticket["agent_response"] == "We can help."
    assert ticket["category"] == "PAYMENT"
    assert ticket["intent"] == "request_refund"
    assert ticket["status"] == "resolved"
    assert ticket["tags"] == ["refund", "urgent"]
    assert ticket["source"]["record_id"] == "7"


def test_transform_without_response_creates_new_ticket(bitext_row):
    bitext_row["response"] = ""
    ticket = transform_bitext_record(bitext_row)

    assert ticket["agent_response"] is None
    assert ticket["status"] == "new"


def test_transform_rejects_missing_required_field(bitext_row):
    del bitext_row["intent"]

    with pytest.raises(ValueError, match="intent"):
        transform_bitext_record(bitext_row)


def test_transform_records_assigns_source_indices(bitext_row):
    tickets = transform_bitext_records([bitext_row, dict(bitext_row)])

    assert [ticket["source"]["record_id"] for ticket in tickets] == ["0", "1"]
