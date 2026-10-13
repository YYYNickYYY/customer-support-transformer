"""Read-only checks of phase freezes, parity, budgets and exploratory probe summaries."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reply-prefix-20261006-v1"
pytestmark = pytest.mark.skipif(not (OUT / "assessment-freeze.json").exists(), reason="Local prefix diagnostic absent")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def replies():
    return [json.loads(line) for line in (OUT / "replies.jsonl").read_text(encoding="utf-8").splitlines()]


def test_all_phases_unchanged_and_no_updates():
    for name in ("freeze.json", "execution-freeze.json", "assessment-freeze.json"):
        for relative, expected in load(name)["sha256"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    assert load("freeze.json")["stage"] == "before_inference"
    assert load("execution-summary.json")["optimizer_updates"] == 0
    assert load("execution-summary.json")["model_gradients_none"]
    assert load("execution-summary.json")["model_state_unchanged"]
    assert load("plan.json")["training"] is False


def test_all_baselines_and_identical_control():
    previous = ROOT / "outputs/condition-use-20261006-v1/replies.jsonl"
    prior = {v["id"]: v for v in [json.loads(line) for line in previous.read_text(encoding="utf-8").splitlines()]}
    rows = replies()
    assert len(rows) == len({v["id"] for v in rows}) == 25
    assert load("parity.json")["matched"] == 8
    for row in rows[:8]:
        old = prior[row["original_id"]]
        assert all(row[key] == old[key] for key in ("response", "generated_token_ids", "stop_reason"))
    assert all(rows[8][key] == rows[-1][key] for key in ("response", "generated_token_ids", "stop_reason"))


def test_total_budget_and_trace_integrity():
    plan = load("plan.json")
    cfg = plan["generation"]["model_config"]
    blocked = {cfg[f"{name}_token_id"] for name in ("pad", "bos", "user", "assistant")}
    for row in replies():
        trace = load("traces/" + row["id"] + ".json")
        canonical = json.dumps(trace["payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(canonical.encode()).hexdigest() == trace["sha256"]
        assert row["generated_tokens"] == row["prefix_tokens"] + row["continuation_tokens"] <= 320
        assert row["generated_token_ids"] == row["prefix_token_ids"] + row["continuation_token_ids"]
        assert row["prompt_tokens"] + row["generated_tokens"] <= 512
        assert len(trace["payload"]["steps"]) == row["continuation_tokens"]
        assert not any(i in blocked for i in row["generated_token_ids"])
        assert cfg["eos_token_id"] not in row["prefix_token_ids"]
        if row["stop_reason"] == "eos":
            assert row["generated_token_ids"][-1] == cfg["eos_token_id"]


def test_trajectory_prefixes_exact_and_score_arithmetic():
    from my_project_1.reply_prefix import trajectory_prefixes
    from my_project_1.tokenization import load_frozen
    plan = load("plan.json")
    tokenizer, _ = load_frozen(ROOT / plan["generation"]["model_config"]["tokenizer_artifact"])
    baselines = {r["original_id"]: r for r in replies()[:8]}
    probes = load("trajectory-probes.json")
    assert len(probes) == 32 and sum(p["available"] for p in probes) == 31
    for source, row in baselines.items():
        expected = trajectory_prefixes(tokenizer, row["generated_token_ids"], 4)
        actual = [p for p in probes if p["source_original_id"] == source]
        assert all({k: p[k] for k in stage} == stage for p, stage in zip(actual, expected))
        for p in actual:
            if not p["available"]:
                assert p["question_scores"] is None and p["condition_separation"] is None
                continue
            a, b = p["pair"]
            fa, fb = a.split("-", 1)[0], b.split("-", 1)[0]
            scores = p["question_scores"]
            for question in (a, b):
                branches = scores[question]["branches"]
                assert scores[question]["margin_A_over_B"] == branches[fb]["mean_nll"] - branches[fa]["mean_nll"]
                for branch in branches.values():
                    assert len(branch["per_token_nll"]) == branch["token_count"] == len(branch["target_ids"])
                    assert branch["total_nll"] == pytest.approx(sum(branch["per_token_nll"]), abs=1e-4)
                    assert branch["mean_nll"] == pytest.approx(branch["total_nll"] / branch["token_count"], abs=1e-5)
            assert p["condition_separation"] == scores[a]["margin_A_over_B"] - scores[b]["margin_A_over_B"]


def test_full_text_evidence_and_summary_denominators():
    rows = {r["id"]: r for r in replies()}
    assessments = load("assessments.json")
    assert len(assessments) == len({a["id"] for a in assessments}) == 25
    for a in assessments:
        row = rows[a["id"]]
        assert a["action_match"] == (a["verdict"] == "match")
        assert all(e in row["response"] for e in a["evidence"])
        if a["prefix_condition"] in ("neutral", "task"):
            assert all(e in row["continuation"] for e in a["evidence"])
    summary = load("assessment-summary.json")
    assert summary["forced_prefix_causal_task_interpretation_valid"] is False
    assert summary["forced_prefix_records"] == 17
    for family, conditions in summary["metrics"].items():
        for condition, expected in conditions.items():
            selected = [a for a in assessments if a["family"] == family and a["prefix_condition"] == condition]
            assert expected == {"n": len(selected), "matched": sum(a["action_match"] for a in selected),
                                "verdicts": dict(Counter(a["verdict"] for a in selected))}


def test_trajectory_summary_recomputed_and_limitations_retained(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    path = ROOT / "scripts/review_reply_prefix.py"
    spec = importlib.util.spec_from_file_location("prefix_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = load("assessment-summary.json")
    assert summary["trajectory"] == module.summarize_probes(load("trajectory-probes.json"))
    assert summary["trajectory"]["numbered_marker_stage_records"] == 5
    assert summary["trajectory"]["later_separation_below_zero_stage"] == 23
    assert summary["trajectory"]["all_available_separation_positive"]
