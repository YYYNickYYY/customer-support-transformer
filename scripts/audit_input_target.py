"""Freeze before scan; create-only outputs, no training or generation."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from datetime import datetime, timezone

from my_project_1.input_target_audit import QUOTAS, VERSION, select

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/train-input-target-audit-20261006-v1"
TRAIN = "data/derived/refund-execution-v3.1/train.jsonl"
RUN = "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537"
EXPECTED = "4d1ed247e99b7b9aa5914be35ea9e6bcb8d4a8f70f1188808cda5b8bd07fbe73"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def create(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def register():
    if OUT.exists():
        raise FileExistsError("Registration directory already exists; use verify, never overwrite")
    exp = read(ROOT / RUN / "experiment.json")
    manifest_path = exp["train_revision"]["manifest_path"].replace("\\", "/")
    assert manifest_path == "data/derived/refund-execution-v3.1/manifest.json"
    assert exp["actual_train_sha256"] == EXPECTED == digest(ROOT / TRAIN)
    assert digest(ROOT / manifest_path) == exp["train_revision"]["manifest_sha256"]
    paths = [TRAIN, manifest_path, RUN + "/experiment.json", RUN + "/config.json",
             "artifacts/tokenizer/bitext-bpe-v1/tokenizer.json",
             "docs/train-input-target-audit-protocol-v1.md",
             "src/my_project_1/input_target_audit.py", "scripts/audit_input_target.py",
             "tests/test_input_target_audit.py",
             "outputs/input-sensitivity-20261006-v1/plan.json",
             "outputs/input-sensitivity-20261006-v1/assessments.json",
             "outputs/input-sensitivity-20261006-v1/assessment-summary.json",
             "docs/evaluation-rubric-v1.md", "docs/evaluation-risk-rubric-v1.1.md"]
    # Hash-only protection for historical workbooks, not spreadsheet processing.
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / "outputs").rglob("*.xlsx")]
    hashes = {p: digest(ROOT / p) for p in sorted(set(paths))}
    OUT.mkdir(parents=True)
    create(OUT / "plan.json", {"version": VERSION, "actual_train": TRAIN,
           "actual_train_sha256": EXPECTED, "run": RUN, "quotas": QUOTAS,
           "ranking": "SHA256(train-input-target-audit-v1:42:<stratum>:<ID>), ID tiebreak",
           "semantics": "manual full-text review; missing=pending; lexical hits are not defects",
           "source_scope": "actual v3.1 train only; no dev/sealed_test/original train",
           "training": False, "source_edits": False})
    hashes[(OUT / "plan.json").relative_to(ROOT).as_posix()] = digest(OUT / "plan.json")
    create(OUT / "preregistration-freeze.json", {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "before_train_content_scan", "python": platform.python_version(),
        "files_sha256": hashes})
    print("Registered before content scan; source linkage verified; no training.")


def verify():
    frozen = read(OUT / "preregistration-freeze.json")
    for relative, expected in frozen["files_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    return frozen


def scan():
    verify()
    if (OUT / "scan.json").exists():
        raise FileExistsError("Scan already exists; use verify")
    rows = [json.loads(line) for line in (ROOT / TRAIN).read_text(encoding="utf-8").splitlines()]
    result = select(rows)
    by_id = {r["id"]: r for r in rows}
    samples = [dict(by_id[row_id], strata=[s for s, ids in result["selected_ids"].items()
                                         if row_id in ids], review_status="pending")
               for row_id in result["unique_ids"]]
    create(OUT / "scan.json", result)
    create(OUT / "samples.json", samples)
    create(OUT / "scan-freeze.json", {"scanned_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_sha256": {name: digest(OUT / name) for name in ("scan.json", "samples.json")},
        "semantic_reviews": "pending"})
    print(json.dumps({"rows": result["rows"], "unique_selected": len(samples),
                      "strata": result["strata"],
                      "lexical": result["lexical_identifier_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "scan", "verify"))
    args = parser.parse_args()
    if args.stage == "register":
        register()
    elif args.stage == "scan":
        scan()
    else:
        verify()
        if (OUT / "scan-freeze.json").exists():
            for name, expected in read(OUT / "scan-freeze.json")["files_sha256"].items():
                assert digest(OUT / name) == expected
        print("Frozen dependencies and scan artifacts verified unchanged.")
