"""Profile the actual Bitext CSV file at a pinned dataset revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

DATASET_REPOSITORY = (
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)
DATASET_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
REQUIRED_COLUMNS = {"instruction", "category", "intent"}
WORD_PATTERN = re.compile(r"\S+")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file hash without loading the whole dataset into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: list[int], percentile: float) -> int:
    """Return a deterministic nearest-rank percentile from non-empty values."""

    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def length_summary(values: list[int], *, unit: str, definition: str) -> dict[str, Any]:
    """Summarize a non-empty sequence of integer lengths."""

    return {
        "unit": unit,
        "definition": definition,
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 3),
        "median": median(values),
        "p95_nearest_rank": nearest_rank(values, 0.95),
    }


def profile_csv(path: Path, *, revision: str) -> dict[str, Any]:
    """Compute values from CSV rows rather than trusting dataset-card claims."""

    categories: Counter[str] = Counter()
    intents: Counter[str] = Counter()
    category_intents: dict[str, set[str]] = defaultdict(set)
    character_lengths: list[int] = []
    word_lengths: list[int] = []
    missing = Counter()
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        absent = REQUIRED_COLUMNS.difference(columns)
        if absent:
            raise ValueError(f"missing required columns: {', '.join(sorted(absent))}")

        for row in reader:
            row_count += 1
            instruction = row.get("instruction") or ""
            category = row.get("category") or ""
            intent = row.get("intent") or ""

            for field, value in (
                ("instruction", instruction),
                ("category", category),
                ("intent", intent),
            ):
                if not value:
                    missing[field] += 1

            if category:
                categories[category] += 1
            if intent:
                intents[intent] += 1
            if category and intent:
                category_intents[category].add(intent)

            character_lengths.append(len(instruction))
            word_lengths.append(len(WORD_PATTERN.findall(instruction)))

    if not row_count:
        raise ValueError("dataset contains no rows")

    return {
        "source": {
            "repository": DATASET_REPOSITORY,
            "revision": revision,
            "file": DATASET_FILE,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "row_count": row_count,
        "columns": columns,
        "missing_values": {field: missing[field] for field in sorted(REQUIRED_COLUMNS)},
        "category": {
            "unique_count": len(categories),
            "values": sorted(categories),
            "counts": dict(sorted(categories.items())),
        },
        "intent": {
            "unique_count": len(intents),
            "values": sorted(intents),
            "counts": dict(sorted(intents.items())),
        },
        "category_intents": {
            category: sorted(values)
            for category, values in sorted(category_intents.items())
        },
        "instruction_length": {
            "characters": length_summary(
                character_lengths,
                unit="Unicode code points",
                definition="len(instruction) after CSV decoding; no trimming or normalization",
            ),
            "whitespace_words": length_summary(
                word_lengths,
                unit="non-whitespace spans",
                definition="count of regex matches for \\S+ in the decoded instruction",
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    profile = profile_csv(args.csv_path, revision=args.revision)
    actual_hash = profile["source"]["sha256"]
    if args.expected_sha256 and actual_hash.lower() != args.expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch: expected {args.expected_sha256}, got {actual_hash}"
        )

    rendered = json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
