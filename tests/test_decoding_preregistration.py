"""Read-only preregistration checks; intentionally never import torch."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/decoding-repetition-preregister-20261006-v1"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_plan_is_not_an_active_generation_or_training_config():
    plan = load("plan.json")
    assert plan["status"] == "preregistered_not_run"
    assert plan["implementation_status"] == "not_implemented"
    assert plan["results"] is None and plan["entity_ratings"] is None
    assert plan["generation_runs"] == plan["optimizer_updates"] == 0
    assert not plan["training_activation"] and not plan["sealed_test_access"]
    assert (plan["control"], plan["candidate"]) == (0, 4)
    assert plan["constraint_scope"] == "generated_reply_token_ids_only"
    assert plan["eos_exempt"] and not plan["entity_exemption"]


def test_guardrails_and_frozen_scope_are_complete():
    plan = load("plan.json")
    freeze = load("freeze.json")
    assert len(freeze["ordered_ids"]) == len(set(freeze["ordered_ids"])) == 135
    assert set(plan["guardrails"]) == {"intent", "helpfulness", "fluency", "risk_v1.1", "entity_audit", "truncation"}
    assert plan["acceptance"]["no_new_risk_or_entity_error_per_id"]
    assert not plan["acceptance"]["automatic_adoption"]
    assert freeze["checks"]["baseline_repetition"] == 63
    assert freeze["checks"]["historical_tables_unchanged"] == 13


def test_registration_repeat_is_idempotent_and_does_not_edit_sources():
    before = (OUT / "freeze.json").read_bytes()
    result = subprocess.run([sys.executable, "-X", "utf8", "scripts/run_decoding_comparison.py", "verify"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert "no inference" in result.stdout
    assert (OUT / "freeze.json").read_bytes() == before


def test_frozen_files_still_match():
    for name, expected in load("freeze.json")["source_sha256"].items():
        digest = hashlib.sha256()
        with (ROOT / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected, name
