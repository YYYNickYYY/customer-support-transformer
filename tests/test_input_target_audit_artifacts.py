"""Verify the local frozen audit, skip on clones without derived artifacts."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/train-input-target-audit-20261006-v1"
pytestmark = pytest.mark.skipif(not (OUT / "review-freeze.json").exists(), reason="Local audit artifacts absent")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preregistration_and_phase_freezes():
    registered = load("preregistration-freeze.json")
    assert registered["stage"] == "before_train_content_scan"
    assert registered["registered_at_utc"] < load("scan-freeze.json")["scanned_at_utc"]
    for name, expected in registered["files_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for freeze in ("scan-freeze.json", "review-freeze.json"):
        for name, expected in load(freeze)["files_sha256"].items():
            assert sha(OUT / name) == expected, name
    assert sha(ROOT / "scripts/finalize_input_target_audit.py") == load("review-freeze.json")["review_script_sha256"]
    assert sha(ROOT / "docs/train-input-target-audit-20261006.md") == load("review-freeze.json")["report_sha256"]


def test_complete_review_and_literal_evidence():
    samples = {r["id"]: r for r in load("samples.json")}
    reviews = load("reviews.json")
    assert len(reviews) == len(samples) == len({r["id"] for r in reviews}) == 48
    for r in reviews:
        assert r["evidence"] in samples[r["id"]]["response"]
        assert r["strata"] == samples[r["id"]]["strata"]
        assert r["task_alignment"] in {"aligned", "mismatch", "ambiguous", "not_applicable"}
        assert r["reason"]


def test_recomputed_stratum_denominators_and_empty_not_passed():
    summary = load("review-summary.json")
    for key, ids in load("scan.json")["selected_ids"].items():
        subset = [r for r in load("reviews.json") if r["id"] in ids]
        assert summary["strata"][key]["reviewed"] == len(subset)
        assert summary["strata"][key]["task_alignment"] == dict(Counter(r["task_alignment"] for r in subset))
        assert summary["strata"][key]["pending"] == 0
    assert summary["strata"]["direct_negation"]["task_alignment"] == {}
    assert summary["strata"]["explicit_copy"]["reviewed"] == 0


def test_candidate_selection_is_reproducible_from_actual_train_only():
    from my_project_1.input_target_audit import select
    plan = load("plan.json")
    assert plan["training"] is False and plan["source_edits"] is False
    source = ROOT / plan["actual_train"]
    assert sha(source) == plan["actual_train_sha256"]
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    assert select(rows) == load("scan.json")
