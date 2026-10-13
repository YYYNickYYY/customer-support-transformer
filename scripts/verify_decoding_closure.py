"""Validate full paired ratings and materialize input-entity inventories, not new scores."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/decoding-review-closure-20261006-v1/calibrated"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(mapping):
    for path, expected in mapping.items():
        if sha(ROOT / path) != expected:
            raise ValueError("Changed frozen file: " + path)


def save(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("Do not overwrite derived audit")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def input_entities(text):
    pattern = r"\{\{[^{}]+\}\}|\b\d+(?:[.,]\d+)*\b|\b(?:platinum|gold|premium|pro|standard|freemium|dollars|dollras)\b"
    entities = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        value = match.group()
        if value.startswith("{{"):
            role = {"{{Order Number}}": "order_id", "{{Invoice Number}}": "invoice_id",
                    "{{Account Category}}": "account_type", "{{Account Type}}": "account_type",
                    "{{Refund Amount}}": "refund_amount", "{{Person Name}}": "invoice_person",
                    "{{Delivery City}}": "delivery_city", "{{Delivery Country}}": "delivery_country"}.get(value, "unclassified_placeholder")
        elif value[0].isdigit():
            role = "invoice_id" if "invoice" in text.lower() else "order_id"
        elif value.lower() in ("dollars", "dollras"):
            role = "currency_dollar_word"
        else:
            role = "account_type"
        entities.append({"text": value, "start": match.start(), "end": match.end(), "role": role})
    return entities


def main():
    base = OUT.parent
    generation = read(ROOT / "outputs/decoding-repetition-comparison-20261006-v1/generation-freeze.json")
    verify(generation["sha256"])
    verify(generation["dependencies_sha256"])
    raw = read(base / "final-freeze.json")
    verify(raw["sha256"])
    verify(raw["source_sha256"])
    final = read(OUT / "freeze.json")
    verify(final["sources_sha256"])
    verify(final["files_sha256"])
    ratings = read(OUT / "ratings.json")
    queue = read(base / "queue.json")
    source = {blind: row for row in queue for blind in row["blind_ids"]}
    if len(ratings) != 270 or len({(v["id"], v["arm"]) for v in ratings}) != 270:
        raise ValueError("Incomplete or duplicate pairs")
    fields = ["intent_score", "helpfulness_score", "fluency_score", "repetition", "unsupported_claim"]
    entities = []
    for row in ratings:
        q = source[row["blind_id"]]
        if any(type(row[k]) is not int or not 0 <= row[k] <= (2 if i < 3 else 1) for i, k in enumerate(fields)):
            raise ValueError("Invalid score")
        if not row["notes"] or not row["customer_request"] or not row["evidence"]:
            raise ValueError("Missing full-text evidence")
        if any(e not in q["instruction"] and e not in q["response"] for e in row["evidence"]):
            raise ValueError("Non-verbatim evidence")
        entities.append({"id": row["id"], "arm": row["arm"], "input_entities": input_entities(q["instruction"]),
            "output_placeholder_occurrences": row["entity_audit"]["output_placeholder_occurrences"],
            "human_ai_error_tags": row["entity_audit"]["error_tags"], "notes": row["notes"],
            "evidence": row["evidence"], "no_automatic_entity_score": True})
    metrics = read(OUT / "summary.json")["metrics_sums"]
    for scope, arms in metrics.items():
        for arm, totals in arms.items():
            selected = [v for v in ratings if v["arm"] == arm and
                (scope == "all135" or v["cohort"] == scope or "intent:" + v["intent"] == scope)]
            expected = {"n": len(selected), **{k: sum(v[k] for v in selected) for k in fields},
                        "entity_errors": sum(v["entity_audit"]["has_error"] for v in selected)}
            if expected != totals:
                raise ValueError("Summary mismatch " + scope)
    inventory = read(ROOT / "outputs/table-management-20261006/historical-inventory.json")
    verify({v["path"]: v["sha256"] for v in inventory})
    save(OUT / "entity-audit.json", entities)
    paths = [OUT / "entity-audit.json", OUT / "freeze.json", Path(__file__).resolve()]
    save(OUT / "validation.json", {"status": "passed", "ratings": 270, "pairs": 135,
        "unique_text_reviews": len(queue), "historical_tables_unchanged": len(inventory),
        "summary_scopes": len(metrics), "verbatim_evidence_checked": True,
        "entity_inventory_includes_account_type_currency_and_numeric_ids": True,
        "sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}})
    print("270 ratings / 135 pairs verified; all 30 scopes recomputed; 13 old tables unchanged.")


if __name__ == "__main__":
    main()
