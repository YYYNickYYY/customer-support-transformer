"""Read-only lexical screening and fixed input-based sampling, not semantic scoring."""
import hashlib
import re
from collections import Counter

VERSION = "train-input-target-audit-v1"
ID_PATTERN = re.compile(r"\b(order|invoice)\s*(?:(?:number|no\.?|id)\s*)?[#:]?\s*(\d{3,})\b", re.I)
NEGATIVE = re.compile(r"\b(?:do not|don't|not|cannot|can't|no longer)\b", re.I)
DIRECT = re.compile(r"\b(?:do not|don't)\s+(?:want|wish)\s+to\s+(?:cancel\b|(?:request|get|apply for)\s+(?:a\s+)?refund\b)", re.I)
PARTIAL = re.compile(r"\b(?:one|some|several|single|part|an? item)\b", re.I)
REFUND_INTENTS = {"get_refund", "check_refund_policy", "track_refund"}
QUOTAS = dict.fromkeys(("identifiers", "explicit_copy", "direct_negation", "broad_negation", "partial_items", "refund_steps"), 8)
QUOTAS.update(dict.fromkeys(("identifiers_control", "negation_control", "partial_control", "refund_control"), 4))


def identifiers(text):
    return [{"kind": m[1].lower(), "value": m[2]} for m in ID_PATTERN.finditer(text)]


def signals(row):
    text = row["instruction"]
    intent = row["intent"]
    ids = identifiers(text)
    neg = bool(NEGATIVE.search(text))
    partial = bool(re.search(r"\b(?:remove|delete)\b", text, re.I) and
                   re.search(r"\b(?:items?|products?)\b", text, re.I) and PARTIAL.search(text))
    refund = bool(re.search(r"\brefund\b", text, re.I) and
                  re.search(r"\b(?:how|steps?|request|apply)\b", text, re.I) and intent in REFUND_INTENTS)
    return {
        "identifiers": bool(ids),
        "explicit_copy": bool(ids and re.search(r"\b(?:include|repeat|mention|copy)\b", text, re.I)),
        "direct_negation": bool(DIRECT.search(text)),
        "broad_negation": neg,
        "partial_items": partial,
        "refund_steps": refund,
        "identifiers_control": bool(re.search(r"\b(?:order|invoice)\b", text, re.I) and not ids),
        "negation_control": intent in {"cancel_order", "get_refund"} and not neg,
        "partial_control": intent in {"change_order", "cancel_order"} and not partial,
        "refund_control": intent in REFUND_INTENTS and not refund,
    }


def select(rows):
    seen = set()
    pools = {key: [] for key in QUOTAS}
    lexical = Counter()
    for row in rows:
        row_id = row["id"]
        if row_id in seen:
            raise ValueError("Duplicate ID")
        seen.add(row_id)
        for key, hit in signals(row).items():
            if hit:
                pools[key].append(row)
        ids = identifiers(row["instruction"])
        if ids:
            lexical["identifier_rows"] += 1
            lexical["all_values_retained"] += all(
                re.search(r"(?<!\d)" + re.escape(item["value"]) + r"(?!\d)", row["response"])
                for item in ids)
            lexical["target_has_order_invoice_placeholder"] += bool(re.search(
                r"\{\{(?:Order|Invoice)\s+(?:Number|ID)\}\}", row["response"], re.I))
    selected, counts = {}, {}
    for key, pool in pools.items():
        ranked = sorted(pool, key=lambda r: (
            hashlib.sha256(f'{VERSION}:42:{key}:{r["id"]}'.encode()).hexdigest(), r["id"]))
        selected[key] = [r["id"] for r in ranked[:QUOTAS[key]]]
        counts[key] = {"candidates": len(pool), "quota": QUOTAS[key],
                       "selected": len(selected[key]), "shortfall": max(0, QUOTAS[key] - len(pool)),
                       "intent_counts": dict(sorted(Counter(r["intent"] for r in pool).items()))}
    unique = sorted({row_id for ids in selected.values() for row_id in ids})
    return {"rows": len(seen), "strata": counts, "selected_ids": selected,
            "unique_ids": unique, "lexical_identifier_counts": dict(lexical)}
