"""Read-only acceptance of saved measurements; never load the fifth model or regenerate."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

from my_project_1.tokenization import load_frozen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/reply-prefix-20261006-v1"
OUT = ROOT / "outputs/reply-prefix-measurement-acceptance-20261006-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def recompute_probe(probe):
    if not probe["available"]:
        assert probe["question_scores"] is None and probe["condition_separation"] is None
        return
    a, b = probe["pair"]
    fa, fb = a.split("-", 1)[0], b.split("-", 1)[0]
    scores = probe["question_scores"]
    for question in (a, b):
        branches = scores[question]["branches"]
        for value in branches.values():
            n = value["token_count"]
            assert n == len(value["target_ids"]) == len(value["per_token_nll"]) > 0
            assert abs(sum(value["per_token_nll"]) - value["total_nll"]) < 1e-4
            assert abs(value["total_nll"] / n - value["mean_nll"]) < 1e-5
        assert scores[question]["margin_A_over_B"] == branches[fb]["mean_nll"] - branches[fa]["mean_nll"]
    assert probe["condition_separation"] == scores[a]["margin_A_over_B"] - scores[b]["margin_A_over_B"]
    sign = 1 if probe["source_original_id"] == a else -1
    assert probe["own_correct_margin"] == sign * scores[probe["source_original_id"]]["margin_A_over_B"]


def prefix_boundary(tokenizer, row):
    prefix = row["prefix_token_ids"]
    full_encoded = tokenizer.encode(row["response"], add_special_tokens=False).ids
    piece = tokenizer.id_to_token(prefix[-1]) if prefix else None
    return {"id": row["id"], "prefix_tokens": len(prefix), "last_token_piece": piece,
            "text_ends_whitespace": bool(row["prefix_text"] and row["prefix_text"][-1].isspace()),
            "standalone_whitespace_tail": bool(prefix and piece == "Ġ"),
            "decode_exact": tokenizer.decode(prefix, skip_special_tokens=True) == row["prefix_text"],
            "canonical_full_reply_prefix_matches": full_encoded[:len(prefix)] == prefix,
            "initial_continuation_pieces": [tokenizer.id_to_token(t) for t in row["continuation_token_ids"][:4]]}


def run():
    if OUT.exists():
        raise FileExistsError("Acceptance already exists; do not overwrite")
    paths = [ROOT / "docs/reply-prefix-measurement-acceptance-protocol-v1.md", Path(__file__),
             ROOT / "tests/test_reply_prefix_measurement_acceptance.py"]
    paths += [SOURCE / name for name in ("plan.json", "freeze.json", "execution-freeze.json", "assessment-freeze.json",
        "replies.jsonl", "trajectory-probes.json", "assessments.json", "assessment-summary.json")]
    paths += [ROOT / "src/my_project_1/reply_prefix.py", ROOT / "docs/reply-prefix-protocol-v1.md"]
    OUT.mkdir(parents=True)
    create(OUT / "plan.json", {"scope": "existing outputs, tokenizer and synthetic tests only",
        "generation_rerun": False, "fifth_model_loaded": False, "training": False,
        "overall_gate": "all four layers must pass for task mechanism decisions"})
    paths += [OUT / "plan.json"]
    create(OUT / "freeze.json", {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "before_acceptance_checks", "sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}})
    dependencies = {}
    for name in ("freeze.json", "execution-freeze.json", "assessment-freeze.json"):
        for path, expected in read(SOURCE / name)["sha256"].items():
            assert sha(ROOT / path) == expected, path
            dependencies[path] = expected
    plan = read(SOURCE / "plan.json")
    cfg = plan["generation"]["model_config"]
    tokenizer, _ = load_frozen(ROOT / cfg["tokenizer_artifact"])
    rows = [json.loads(line) for line in (SOURCE / "replies.jsonl").read_text(encoding="utf-8").splitlines()]
    probes = read(SOURCE / "trajectory-probes.json")
    scores = read(SOURCE / "assessments.json")
    summary = read(SOURCE / "assessment-summary.json")
    assert len(rows) == len({r["id"] for r in rows}) == len(scores) == 25
    assert len(probes) == 32
    assert [r["id"] for r in rows] == [r["id"] for r in plan["cases"]]
    previous = ROOT / "outputs/condition-use-20261006-v1/replies.jsonl"
    prior = {r["id"]: r for r in [json.loads(line) for line in previous.read_text(encoding="utf-8").splitlines()]}
    for row in rows[:8]:
        assert all(row[k] == prior[row["original_id"]][k] for k in ("response", "generated_token_ids", "stop_reason"))
    assert all(rows[8][k] == rows[-1][k] for k in ("response", "generated_token_ids", "stop_reason"))
    assert all(r["generated_tokens"] == r["prefix_tokens"] + r["continuation_tokens"] <= 320 for r in rows)
    assert all("response" not in case and "labels" not in case for case in plan["cases"])
    for probe in probes:
        recompute_probe(probe)
    for family, conditions in summary["metrics"].items():
        for condition, expected in conditions.items():
            selected = [s for s in scores if s["family"] == family and s["prefix_condition"] == condition]
            assert expected == {"n": len(selected), "matched": sum(s["action_match"] for s in selected),
                                "verdicts": dict(Counter(s["verdict"] for s in selected))}
    boundaries = [prefix_boundary(tokenizer, row) for row in rows if row["prefix_condition"] != "empty"]
    markers = [{"source": p["source_original_id"], "stage": p["stage"], "prefix_tokens": p["prefix_tokens"],
                "prefix_tail": p["prefix_text"][-100:]} for p in probes if p["available"] and
               re.search(r"(?:^|\n)\s*\d+\.$", p["prefix_text"])]
    agreements = []
    judged = {s["id"].removesuffix("-empty"): s for s in scores if s["prefix_condition"] == "empty"}
    for p in probes:
        if p["available"] and p["own_correct_margin"] < 0 and judged[p["source_original_id"]]["action_match"]:
            agreements.append({"source": p["source_original_id"], "stage": p["stage"],
                "own_correct_margin": p["own_correct_margin"], "full_reply_task_match": True,
                "meaning": "counterexample to reading margin sign as full-reply correctness, not a defined next-action gold error"})
    branches = {family: {"text": text, "token_count": len(tokenizer.encode(text, add_special_tokens=False).ids)}
                for family, text in plan["branches"].items()}
    result = {"status": "not_accepted_for_task_mechanism_or_training_decisions",
        "layers": {"provenance_execution": "pass", "saved_score_arithmetic": "pass",
                   "stimulus_boundary": "fail", "task_construct": "not_established", "history_causal_claim": "fail"},
        "provenance_files_verified": len(dependencies), "baselines_verified": 8, "control_verified": True,
        "stage_records": 32, "available": sum(p["available"] for p in probes),
        "forced_prefix_boundary_checks": boundaries, "numbered_list_pseudo_sentence_boundaries": markers,
        "fixed_branch_lengths": branches, "negative_margin_with_correct_full_reply": agreements,
        "judgment_scope": "Arithmetic passes do not establish semantic/causal validity. No fifth logits rerun.",
        "generation_rerun": False, "training": False}
    create(OUT / "acceptance.json", result)
    create(OUT / "acceptance-freeze.json", {"sha256": {
        (OUT / name).relative_to(ROOT).as_posix(): sha(OUT / name) for name in ("plan.json", "freeze.json", "acceptance.json")}})
    print(json.dumps({"status": result["status"], "layers": result["layers"],
        "standalone_space_prefixes": sum(b["standalone_whitespace_tail"] for b in boundaries),
        "canonical_prefix_mismatches": sum(not b["canonical_full_reply_prefix_matches"] for b in boundaries),
        "pseudo_sentence_boundaries": len(markers), "branch_lengths": {k: v["token_count"] for k, v in branches.items()},
        "counterexamples": len(agreements)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
