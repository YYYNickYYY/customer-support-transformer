import json
from pathlib import Path


SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "work-order.schema.json"


def test_work_order_schema_is_valid_json_and_declares_core_fields():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["additionalProperties"] is False
    assert {
        "ticket_id",
        "customer_message",
        "category",
        "intent",
        "status",
        "source",
    } <= set(schema["required"])
