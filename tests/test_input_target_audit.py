import pytest
from my_project_1.input_target_audit import identifiers, select, signals


def row(text, target="Please let us check.", intent="get_refund", row_id="a"):
    return dict(id=row_id, instruction=text, response=target, intent=intent)


def test_context_identifiers_and_leading_zero():
    assert identifiers("invoice #00108, order number 85632") == [
        {"kind": "invoice", "value": "00108"}, {"kind": "order", "value": "85632"}]
    assert identifiers("Pay $100 on 2026-10-06; step 123") == []
    assert identifiers("order {{Order Number}}") == []


@pytest.mark.parametrize("text,expected", [
    ("I do not want to cancel my order", True),
    ("I don't want to request a refund", True),
    ("I don't know how to request a refund", False),
    ("I cannot pay", False),
    ("I no longer want newsletters", False),
])
def test_negation_is_only_a_screen(text, expected):
    assert signals(row(text))["direct_negation"] is expected


def test_partial_and_refund_screen():
    assert signals(row("remove one item from order", intent="change_order"))["partial_items"]
    assert not signals(row("cancel whole order", intent="change_order"))["partial_items"]
    assert signals(row("how can I request a refund"))["refund_steps"]
    assert not signals(row("check my refund status", intent="track_refund"))["refund_steps"]


def test_selection_is_order_independent_and_target_blind():
    rows = [row("how to request a refund", row_id=str(i)) for i in range(20)]
    assert select(rows) == select(list(reversed(rows)))
    changed = [dict(r, response="Wrong guarantee") for r in rows]
    assert select(rows)["selected_ids"] == select(changed)["selected_ids"]
    assert len(select(rows)["selected_ids"]["refund_steps"]) == 8
    with pytest.raises(ValueError, match="Duplicate"):
        select([rows[0], rows[0]])


def test_lexical_retention_is_not_semantic_scoring():
    result = select([row("include invoice #00108", "Invoice {{Invoice Number}}"),
                     row("order 12345", "Order 123456", row_id="b")])
    assert result["lexical_identifier_counts"]["all_values_retained"] == 0
    assert result["lexical_identifier_counts"]["target_has_order_invoice_placeholder"] == 1
    assert "semantic_score" not in result
