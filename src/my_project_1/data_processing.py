"""Pure data-processing functions for customer-support examples."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

BITEXT_DATASET_NAME = "Bitext Customer Support LLM Chatbot Training Dataset"
_WHITESPACE = re.compile(r"\s+")
_REQUIRED_BITEXT_FIELDS = ("instruction", "category", "intent")


def normalize_text(value: object) -> str:
    """Return a Unicode-normalized, single-line representation of *value*."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return _WHITESPACE.sub(" ", text).strip()


def normalize_tags(value: object) -> list[str]:
    """Normalize a flag string or iterable into sorted, unique tags."""

    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,|;]", value)
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        parts = list(value)
    else:
        parts = [value]

    return sorted({tag for item in parts if (tag := normalize_text(item))})


def stable_ticket_id(record: Mapping[str, Any]) -> str:
    """Build a deterministic ID from the semantic fields in a Bitext row."""

    fingerprint = "\x1f".join(
        normalize_text(record.get(field))
        for field in ("instruction", "response", "category", "intent")
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return f"bitext-{digest}"


def transform_bitext_record(
    record: Mapping[str, Any],
    *,
    row_number: int | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """Convert one Bitext-style row to the work-order schema draft.

    Expected source fields are ``instruction``, ``response``, ``category``,
    ``intent`` and optional ``flags``. The function has no I/O and is therefore
    deterministic and straightforward to unit test.
    """

    missing = [
        field
        for field in _REQUIRED_BITEXT_FIELDS
        if not normalize_text(record.get(field))
    ]
    if missing:
        raise ValueError(f"missing required Bitext fields: {', '.join(missing)}")

    response = normalize_text(record.get("response"))
    record_id = str(row_number) if row_number is not None else None

    return {
        "schema_version": "0.1.0",
        "ticket_id": stable_ticket_id(record),
        "conversation_id": None,
        "created_at": None,
        "updated_at": None,
        "channel": "chat",
        "language": normalize_text(language) or "en",
        "subject": None,
        "customer_message": normalize_text(record["instruction"]),
        "agent_response": response or None,
        "category": normalize_text(record["category"]),
        "intent": normalize_text(record["intent"]),
        "priority": "normal",
        "status": "resolved" if response else "new",
        "sentiment": "unknown",
        "tags": normalize_tags(record.get("flags")),
        "pii_redacted": False,
        "source": {
            "dataset": BITEXT_DATASET_NAME,
            "record_id": record_id,
        },
        "metadata": {},
    }


def transform_bitext_records(
    records: Iterable[Mapping[str, Any]], *, language: str = "en"
) -> list[dict[str, Any]]:
    """Transform Bitext rows while assigning their zero-based source index."""

    return [
        transform_bitext_record(record, row_number=index, language=language)
        for index, record in enumerate(records)
    ]
