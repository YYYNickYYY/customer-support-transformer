"""Read-only validation of the local completed v2 experiment; no model load."""
from collections import Counter

import pytest

from my_project_1 import condition_measurement_v2 as r
from my_project_1.tokenization import load_frozen

pytestmark = pytest.mark.skipif(not (r.OUT / "summary.json").exists(), reason="Local v2 experiment absent")


def test_frozen_results_and_exact_input_reexpansion():
    frozen = r.verify(r.OUT)
    tok, _ = load_frozen(r.ROOT / frozen["config"]["tokenizer_artifact"])
    assert r.expand_jobs(r.load_protocol(), tok, frozen["config"]) == frozen["jobs"]
    for file in ("generation-freeze.json", "review-freeze.json", "score-freeze.json"):
        r.verify_hashes(r.read(r.OUT / file)["sha256"])
    for stage, count in (("generations", 26), ("scores", 148)):
        status = r.read(r.OUT / (stage + "-status.json"))
        assert status["records"] == count and status["controls_pass"]
        assert status["model_state_unchanged"] and status["gradients_none"]
        assert status["optimizer_updates"] == 0


def test_every_record_and_repeat_control_validate_without_inference():
    frozen = r.verify(r.OUT)
    cfg = frozen["config"]
    tok, _ = load_frozen(r.ROOT / cfg["tokenizer_artifact"])
    results = {}
    for kind in ("generations", "scores"):
        rows = {}
        validator = (lambda j, v: r.validate_generation(j, v, tok, cfg)) if kind == "generations" else r.validate_score
        for job in frozen["jobs"][kind]:
            rows[job["id"]] = r.record(r.OUT, kind, job, r.digest(r.OUT / "freeze.json"),
                                      lambda: pytest.fail("Missing frozen record"), validator)
        r.check_controls(frozen["jobs"][kind], rows, kind)
        results[kind] = rows
    r.validate_reviews(r.read(r.OUT / "semantic-reviews.json"), frozen["jobs"]["generations"], results["generations"])


def test_summary_recomputed_independently_and_semantic_counts_exclude_controls():
    summary = r.read(r.OUT / "summary.json")
    assert len(summary["pairs"]) == 12
    for row in summary["pairs"]:
        for contrast in row["contrasts"]:
            wi, hi, pair = contrast["wording"], row["history_index"], row["pair_id"]
            values = {(q, c): r.read(r.OUT / "scores" / f"{pair}-h{hi}-w{wi}-q{q}-c{c}.json")["payload"]["result"]
                      for q in (0, 1) for c in (0, 1)}
            for field, saved in (("mean_nll", "D_mean"), ("total_nll", "D_total")):
                expected = values[(0, 1)][field] - values[(0, 0)][field] - values[(1, 1)][field] + values[(1, 0)][field]
                assert contrast[saved] == pytest.approx(expected, abs=1e-12)
        assert row["classification"] == r.classify([c["D_mean"] for c in row["contrasts"]],
                                                  [c["D_total"] for c in row["contrasts"]])
    reviews = [row for row in r.read(r.OUT / "semantic-reviews.json") if not row["id"].endswith("-repeat")]
    assert len(reviews) == summary["semantic_primary_n"] == 24
    assert summary["semantic_first_action"] == dict(Counter(row["first_action"] for row in reviews))
    assert summary["semantic_full_task"] == dict(Counter(row["full_task"] for row in reviews))
