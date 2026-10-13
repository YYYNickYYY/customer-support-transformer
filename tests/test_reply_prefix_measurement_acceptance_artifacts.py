"""Checks of the separate acceptance ledger without fifth-model inference."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reply-prefix-measurement-acceptance-20261006-v1"
SOURCE = ROOT / "outputs/reply-prefix-20261006-v1"
pytestmark = pytest.mark.skipif(not (OUT / "acceptance.json").exists(), reason="Local acceptance absent")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_acceptance_and_nonmutation_scope():
    for name in ("freeze.json", "acceptance-freeze.json"):
        for relative, expected in load(OUT / name)["sha256"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    result = load(OUT / "acceptance.json")
    assert result["status"] == "not_accepted_for_task_mechanism_or_training_decisions"
    assert result["generation_rerun"] is False and result["training"] is False
    assert load(OUT / "plan.json")["fifth_model_loaded"] is False
    assert result["layers"]["saved_score_arithmetic"] == "pass"
    assert result["layers"]["stimulus_boundary"] == "fail"
    assert result["layers"]["task_construct"] == "not_established"


def test_existing_probabilities_independently_recomputed_and_missing_preserved():
    spec = importlib.util.spec_from_file_location("measurement_accept", ROOT / "scripts/accept_reply_prefix_measurement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probes = load(SOURCE / "trajectory-probes.json")
    for probe in probes:
        module.recompute_probe(probe)
    result = load(OUT / "acceptance.json")
    assert result["stage_records"] == len(probes) == 32
    assert result["available"] == sum(p["available"] for p in probes) == 31
    assert len(result["numbered_list_pseudo_sentence_boundaries"]) == 5
    assert len(result["negative_margin_with_correct_full_reply"]) == 4


def test_boundary_findings_are_limited_not_all_encoding_errors():
    result = load(OUT / "acceptance.json")
    rows = result["forced_prefix_boundary_checks"]
    assert len(rows) == 17 and all(r["standalone_whitespace_tail"] for r in rows)
    assert all(r["decode_exact"] for r in rows)
    assert sum(not r["canonical_full_reply_prefix_matches"] for r in rows) == 1
    assert {k: v["token_count"] for k, v in result["fixed_branch_lengths"].items()} == {
        "partial": 12, "whole": 13, "request": 13, "policy": 15}
