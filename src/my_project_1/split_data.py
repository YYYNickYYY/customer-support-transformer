"""Pinned Bitext group split; existing frozen output is never overwritten."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re

REVISION = "fb86d1b60038970d79fefbfa0c28b47c22d8b961"
SOURCE_SHA256 = "6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434"
NAMES = ("train", "dev", "sealed_test")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def template_key(instruction: str) -> str:
    """Remove {{entity}} spans, lowercase, collapse whitespace; retain punctuation."""
    return " ".join(re.sub(r"\{\{[^{}]*\}\}", " ", instruction).lower().split())


def split_records(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Assign shuffled, sorted template groups 80/10/10 (by group count).

    IDs identify source rows within the pinned revision, including duplicates.
    Template IDs depend only on instruction, never category/intent/response.
    """
    groups = {}
    for index, row in enumerate(rows):
        if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
            raise ValueError(f"Missing instruction at source row {index}")
        key = template_key(row["instruction"])
        groups.setdefault(key, []).append({
            **row, "id": f"bitext:{REVISION}:{index}",
            "template_id": digest(key), "source_row": index,
        })
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    if n < 10:
        raise ValueError("At least 10 template groups required for an 80/10/10 split")
    train_end, dev_end = int(n * 0.8), int(n * 0.9)
    result = {}
    for name, selected in zip(NAMES, (keys[:train_end], keys[train_end:dev_end], keys[dev_end:])):
        result[name] = sorted((row for key in selected for row in groups[key]),
                              key=lambda row: row["source_row"])
    return result


def build_split(csv_path: Path, output: Path, seed: int = 42) -> dict:
    raw_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if raw_hash != SOURCE_SHA256:
        raise ValueError("Source SHA-256 does not match the pinned revision")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    splits = split_records(rows, seed)
    # Exclusive directory creation protects frozen IDs on every repeat invocation.
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, records in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = {
        "revision": REVISION, "source_sha256": raw_hash,
        "ids": [r["id"] for r in splits["sealed_test"]],
        "template_ids": sorted({r["template_id"] for r in splits["sealed_test"]}),
    }
    frozen_path = output / "sealed_test_ids.json"
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    files[frozen_path.name] = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    manifest = {
        "revision": REVISION, "source_sha256": raw_hash, "seed": seed,
        "rule": "remove {{...}} spans; lower(); collapse whitespace; preserve punctuation",
        "ratios_by_template_group": [0.8, 0.1, 0.1],
        "algorithm": "sorted keys shuffled with Python random.Random(seed); floor(0.8*n), floor(0.9*n)",
        "id_rule": "bitext:<full revision>:<zero-based CSV record index>",
        "counts": {name: {"rows": len(records),
                           "groups": len({r['template_id'] for r in records})}
                   for name, records in splits.items()},
        "file_sha256": files,
        "sealed_test_policy": "Do not use for training, model selection, or tuning; retain frozen IDs.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/splits/bitext-v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_split(args.csv_path, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
