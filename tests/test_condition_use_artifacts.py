"""Read-only checks of the completed local supported-task diagnostic."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/condition-use-20261006-v1"
pytestmark = pytest.mark.skipif(not (OUT / "assessment-freeze.json").exists(), reason="Local diagnostic absent")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def replies():
    return [json.loads(line) for line in (OUT / "replies.jsonl").read_text(encoding="utf-8").splitlines()]


def test_frozen_dependencies_and_complete_one_pass():
    for name in ("freeze.json", "generation-freeze.json", "assessment-freeze.json"):
        for relative, expected in load(name)["sha256"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    plan = load("plan.json")
    assert plan["training"] is False and plan["no_repeat_ngram_size"] == 0
    assert load("freeze.json")["stage"] == "before_inference"
    assert [v["id"] for v in plan["cases"]] == [v["id"] for v in replies()]
    assert len(replies()) == len({v["id"] for v in replies()}) == 17
    for row in replies():
        trace = load("traces/" + row["id"] + ".json")
        canonical = json.dumps(trace["payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert trace["sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
        assert len(trace["payload"]["steps"]) == row["generated_tokens"] == len(row["generated_token_ids"])
    summary = load("generation-summary.json")
    assert summary["model_state_unchanged"] and summary["model_gradients_none"]
    assert summary["optimizer_updates"] == 0


def test_source_pairs_and_no_reference_leakage():
    from my_project_1.condition_use import build_plan
    audit = ROOT / "outputs/train-input-target-audit-20261006-v1"
    read = lambda name: json.loads((audit / name).read_text(encoding="utf-8"))
    built = build_plan(read("samples.json"), read("reviews.json"))
    plan = load("plan.json")
    assert all(plan[key] == value for key, value in built.items())
    assert all("response" not in c and "intent" not in c for c in plan["cases"])
    assert len(plan["source_rows"]) == 8 and len(plan["pairs"]) == 4


def test_same_input_same_output_and_same_judgment():
    rows = replies()
    for key in ("instruction", "response", "generated_token_ids", "stop_reason"):
        assert rows[0][key] == rows[-1][key]
    scores = {v["id"]: v for v in load("assessments.json")}
    for key in ("verdict", "action_match", "actual_action", "evidence"):
        assert scores[rows[0]["id"]][key] == scores[rows[-1]["id"]][key]


def test_evidence_and_stratified_counts():
    rows = {v["id"]: v for v in replies()}
    assessments = load("assessments.json")
    assert len(assessments) == len({v["id"] for v in assessments}) == 17
    summary = load("assessment-summary.json")
    for item in assessments:
        assert item["action_match"] == (item["verdict"] == "match")
        assert all(q in rows[item["id"]]["response"] for q in item["evidence"])
        assert all(1 <= span["first_token"] <= span["last_token"] <= rows[item["id"]]["generated_tokens"]
                   for span in item["evidence_token_spans"])
    for family, variants in summary["metrics"].items():
        for variant, expected in variants.items():
            selected = [v for v in assessments if v["family"] == family and v["variant"] == variant]
            assert expected == {"n": len(selected), "matched": sum(v["action_match"] for v in selected),
                                "verdicts": dict(Counter(v["verdict"] for v in selected))}


def test_nll_alignment_and_actual_training_visibility():
    from my_project_1.condition_use import matched_target_batches
    from my_project_1.tokenization import load_frozen, prepare_example
    plan = load("plan.json")
    tokenizer, _ = load_frozen(ROOT / plan["generation"]["model_config"]["tokenizer_artifact"])
    by_id = {v["id"]: v for v in plan["cases"]}
    sources = {v["id"]: v for v in plan["source_rows"]}
    losses = load("target-nll.json")
    assert len(losses) == len({v["case_id"] for v in losses}) == 8
    for record in losses:
        own, swapped = by_id[record["case_id"]], by_id[record["swapped_case_id"]]
        target = sources[own["source_id"]]["response"]
        batches, meta = matched_target_batches(tokenizer, own["instruction"], swapped["instruction"], target, 512)
        assert all(record[k] == v for k, v in meta.items())
        labels = [[v for v in batch["labels"] if v != -100] for batch in batches]
        assert labels[0] == labels[1] and len(labels[0]) == record["scored_tokens"]
        original = prepare_example(tokenizer, own["instruction"], target, 512)
        assert original["supervised_tokens"] == record["scored_tokens"]
        assert not original["truncated"]
        assert record["swapped_minus_matched"] == record["swapped_nll"] - record["matched_nll"]
