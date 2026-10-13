"""Prepare a label-hidden unique-text queue; consume explicit AI judgments, not keyword scores."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re

spec = importlib.util.spec_from_file_location("closure_runner", Path(__file__).with_name("run_decoding_comparison.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
ROOT = runner.ROOT
SOURCE = runner.OUT
OUT = ROOT / "outputs/decoding-review-closure-20261006-v1"


def prepare():
    frozen = runner.read(SOURCE / "generation-freeze.json")
    runner.verify_hashes(frozen["sha256"])
    runner.verify_hashes(frozen["dependencies_sha256"])
    rows = runner.read(SOURCE / "review-blinded.json")
    seen, unique = {}, []
    for row in rows:
        key = (row["instruction"], row["response"])
        if key in seen:
            unique[seen[key]]["blind_ids"].append(row["blind_id"])
        else:
            seen[key] = len(unique)
            unique.append({"review_id": row["blind_id"], "blind_ids": [row["blind_id"]],
                           "instruction": row["instruction"], "response": row["response"]})
    runner.save(OUT / "queue.json", unique)
    sources = [SOURCE / "generation-freeze.json", SOURCE / "review-blinded.json", SOURCE / "review-key.json",
               ROOT / "docs/evaluation-rubric-v1.md", ROOT / "docs/evaluation-risk-rubric-v1.1.md",
               ROOT / "docs/decoding-review-closure-protocol-20261006.md", Path(__file__).resolve(), OUT / "queue.json"]
    runner.save(OUT / "review-freeze.json", {"sources_sha256": {p.relative_to(ROOT).as_posix(): runner.digest(p) for p in sources},
        "blind_records": len(rows), "unique_customer_text_pairs": len(unique),
        "method": "AI label-hidden full-text review; prior context known; not independent blind review"})
    print(f"270 rows; {len(unique)} unique question/text pairs; review queue frozen.")


def show(start, count):
    for i, r in enumerate(runner.read(OUT / "queue.json")[start:start + count], start):
        print(f"\n[{i}] {r['review_id']} Q: {r['instruction']}\nR: {r['response']}")


def finalize():
    frozen = runner.read(OUT / "review-freeze.json")
    runner.verify_hashes(frozen["sources_sha256"])
    queue = runner.read(OUT / "queue.json")
    judgments = []
    for path in sorted((OUT / "judgments").glob("*.json")):
        judgments.extend(runner.read(path))
    by_review = {r[0]: r for r in judgments}
    if len(by_review) != len(judgments) or set(by_review) != {r["review_id"] for r in queue}:
        raise ValueError("Missing or duplicate full-text judgments; do not finalize partial review")
    key = {r["blind_id"]: r for r in runner.read(SOURCE / "review-key.json")}
    ratings = []
    fields = ["intent_score", "helpfulness_score", "fluency_score", "repetition", "unsupported_claim"]
    for r in queue:
        j = by_review[r["review_id"]]
        if len(j) != 10 or any(type(x) is not int or not 0 <= x <= (2 if i < 3 else 1) for i, x in enumerate(j[1:6])):
            raise ValueError("Invalid explicit judgment")
        # [id, five scores, entity tags, customer request, rationale, exact quotes]
        if not isinstance(j[6], list) or not j[7] or not j[8] or not j[9]:
            raise ValueError("Missing entity review or evidence")
        for quote in j[9]:
            if quote not in r["response"] and quote not in r["instruction"]:
                raise ValueError("Evidence not verbatim: " + r["review_id"] + " " + quote)
        input_candidates = [{"text": m.group(), "start": m.start(), "end": m.end()}
                            for m in re.finditer(r"\{\{[^{}]+\}\}|\b\d+(?:[.,]\d+)*\b", r["instruction"])]
        output_placeholders = runner.placeholder_occurrences(r["response"])
        for blind_id in r["blind_ids"]:
            ratings.append({**key[blind_id], **dict(zip(fields, j[1:6])), "customer_request": j[7],
                "notes": j[8], "evidence": j[9], "entity_audit": {"input_candidates": input_candidates,
                "output_placeholder_occurrences": output_placeholders, "error_tags": j[6], "has_error": bool(j[6]),
                "method": "full-text AI audit; regex positions only assist, not scoring"},
                "source_review_id": r["review_id"], "identical_text_reuse": len(r["blind_ids"]) > 1,
                "reviewer": frozen["method"], "date": "2026-10-06", "risk_rubric": "v1.1", "other_rubric": "v1.0"})
    ratings.sort(key=lambda r: (r["id"], r["arm"]))
    by_pair = {(r["id"], r["arm"]): r for r in ratings}
    original = runner.read(ROOT / "outputs/repetition-review-20261006-v1/ai-expert-ratings-repetition-revised.json")
    old = {r["id"]: r for r in original["ratings"] if r["model"] == "fifth"}
    pairs, historic = [], []
    for sample_id in sorted(old):
        a, b = by_pair[(sample_id, "control")], by_pair[(sample_id, "candidate")]
        delta = {f: b[f] - a[f] for f in fields}
        pairs.append({"id": sample_id, "intent": a["intent"], "cohort": a["cohort"], "delta": delta,
            "control": {f: a[f] for f in fields}, "candidate": {f: b[f] for f in fields},
            "control_entity_error": a["entity_audit"]["has_error"], "candidate_entity_error": b["entity_audit"]["has_error"],
            "new_risk": a["unsupported_claim"] == 0 and b["unsupported_claim"] == 1,
            "new_entity_error": not a["entity_audit"]["has_error"] and b["entity_audit"]["has_error"],
            "new_zero_intent_or_help": any(a[f] > 0 and b[f] == 0 for f in fields[:2]),
            "control_notes": a["notes"], "candidate_notes": b["notes"]})
        diffs = {f: {"old": old[sample_id][f], "new": a[f]} for f in fields if old[sample_id][f] != a[f]}
        if diffs:
            historic.append({"id": sample_id, "differences": diffs, "new_review_reason": a["notes"],
                             "evidence": a["evidence"], "old_notes": old[sample_id]["notes"]})
    scopes = {"all135": pairs, "original27": [p for p in pairs if p["cohort"] == "original27"],
              "new108": [p for p in pairs if p["cohort"] == "new108"]}
    scopes.update({"intent:" + i: [p for p in pairs if p["intent"] == i] for i in sorted({p["intent"] for p in pairs})})
    metrics = {}
    for scope, rows in scopes.items():
        arms = {}
        for arm in ("control", "candidate"):
            chosen = [by_pair[(p["id"], arm)] for p in rows]
            arms[arm] = {"n": len(chosen), **{f: sum(r[f] for r in chosen) for f in fields},
                         "entity_errors": sum(r["entity_audit"]["has_error"] for r in chosen)}
        metrics[scope] = arms
    aggregate_failures = [scope for scope, arms in metrics.items() if
        any(arms["candidate"][f] < arms["control"][f] for f in fields[:3]) or
        any(arms["candidate"][f] > arms["control"][f] for f in fields[4:] + ["entity_errors"])]
    vetoes = [p for p in pairs if p["new_risk"] or p["new_entity_error"] or p["new_zero_intent_or_help"]]
    runner.save(OUT / "ratings.json", ratings)
    runner.save(OUT / "paired-comparison.json", pairs)
    runner.save(OUT / "historical-rating-differences.json", historic)
    runner.save(OUT / "summary.json", {"status": "complete_270_AI_ratings", "metrics_sums": metrics,
        "paired_repetition": dict(Counter(f"{p['control']['repetition']}->{p['candidate']['repetition']}" for p in pairs)),
        "aggregate_nonregression_failures": aggregate_failures, "per_id_veto_ids": [p["id"] for p in vetoes],
        "adoption_approved": False, "historical_control_differences": len(historic),
        "no_new_generation_or_training": True, "independent_human_review": False})
    paths = list((OUT / "judgments").glob("*.json")) + [OUT / name for name in
        ("ratings.json", "paired-comparison.json", "historical-rating-differences.json", "summary.json", "review-freeze.json")]
    runner.save(OUT / "final-freeze.json", {"sha256": {p.relative_to(ROOT).as_posix(): runner.digest(p) for p in paths},
        "source_sha256": frozen["sources_sha256"], "historical_scores_sha256": runner.digest(ROOT /
        "outputs/repetition-review-20261006-v1/ai-expert-ratings-repetition-revised.json")})
    print(json.dumps(runner.read(OUT / "summary.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "show", "finalize"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    {"prepare": prepare, "show": lambda: show(args.start, args.count), "finalize": finalize}[args.stage]()
