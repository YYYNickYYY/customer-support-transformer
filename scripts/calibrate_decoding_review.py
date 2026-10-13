"""Apply explicit paired calibration decisions, preserving frozen raw review."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location("closure", Path(__file__).with_name("close_decoding_review.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
r = m.runner
BASE = m.OUT
OUT = BASE / "calibrated"


def main():
    f = r.read(BASE / "final-freeze.json")
    r.verify_hashes(f["sha256"])
    r.verify_hashes(f["source_sha256"])
    ratings = r.read(BASE / "ratings.json")
    old_ratings = r.read(BASE / "ratings.json")
    decisions = r.read(BASE / "calibration-decisions.json")
    changes = []
    fields = ["intent_score", "helpfulness_score", "fluency_score", "repetition", "unsupported_claim"]
    for d in decisions["decisions"]:
        if d["field"] not in fields or any(i not in {v["source_review_id"] for v in ratings} for i in d["review_ids"]):
            raise ValueError("Invalid explicit calibration")
        for row in ratings:
            if row["source_review_id"] in d["review_ids"]:
                changes.append({"id": row["id"], "arm": row["arm"], "field": d["field"],
                                "old": row[d["field"]], "new": d["score"], "reason": d["reason"]})
                row[d["field"]] = d["score"]
                row.setdefault("calibration_notes", []).append(d["reason"])
    by_pair = {(v["id"], v["arm"]): v for v in ratings}
    original = r.read(r.ROOT / "outputs/repetition-review-20261006-v1/ai-expert-ratings-repetition-revised.json")
    history = {v["id"]: v for v in original["ratings"] if v["model"] == "fifth"}
    pairs, differences = [], []
    for sid in sorted(history):
        a, b = by_pair[(sid, "control")], by_pair[(sid, "candidate")]
        new_tags = sorted(set(b["entity_audit"]["error_tags"]) - set(a["entity_audit"]["error_tags"]))
        pairs.append({"id": sid, "intent": a["intent"], "cohort": a["cohort"],
            "control": {k: a[k] for k in fields}, "candidate": {k: b[k] for k in fields},
            "delta": {k: b[k] - a[k] for k in fields},
            "new_entity_error_tags": new_tags,
            "control_entity_errors": a["entity_audit"]["error_tags"], "candidate_entity_errors": b["entity_audit"]["error_tags"],
            "new_risk": not a["unsupported_claim"] and bool(b["unsupported_claim"]),
            "new_zero_intent_or_help": any(a[k] > 0 and b[k] == 0 for k in fields[:2]),
            "control_evidence": a["evidence"], "candidate_evidence": b["evidence"],
            "control_notes": a["notes"], "candidate_notes": b["notes"],
            "calibration_notes": a.get("calibration_notes", []) + b.get("calibration_notes", [])})
        diff = {k: {"old": history[sid][k], "new": a[k]} for k in fields if history[sid][k] != a[k]}
        if diff:
            differences.append({"id": sid, "difference": diff, "old_notes": history[sid]["notes"],
                                "new_notes": a["notes"], "calibration": a.get("calibration_notes", []), "evidence": a["evidence"]})
    scopes = {"all135": pairs, "original27": [p for p in pairs if p["cohort"] == "original27"],
              "new108": [p for p in pairs if p["cohort"] == "new108"]}
    scopes.update({"intent:" + i: [p for p in pairs if p["intent"] == i] for i in sorted({p["intent"] for p in pairs})})
    metrics = {scope: {arm: {"n": len(rows), **{k: sum(p[arm][k] for p in rows) for k in fields},
                           "entity_errors": sum(bool(p[arm + "_entity_errors"]) for p in rows)}
                      for arm in ("control", "candidate")} for scope, rows in scopes.items()}
    regressions = [p for p in pairs if any(p["delta"][k] < 0 for k in fields[:3]) or
                   any(p["delta"][k] > 0 for k in fields[3:]) or p["new_entity_error_tags"]]
    improvements = [p for p in pairs if any(p["delta"][k] > 0 for k in fields[:3]) or
                    any(p["delta"][k] < 0 for k in fields[3:]) or
                    set(p["control_entity_errors"]) - set(p["candidate_entity_errors"])]
    aggregate_failures = [scope for scope, arms in metrics.items() if
        any(arms["candidate"][k] < arms["control"][k] for k in fields[:3]) or
        any(arms["candidate"][k] > arms["control"][k] for k in fields[4:] + ["entity_errors"])]
    vetoes = [p["id"] for p in pairs if p["new_risk"] or p["new_entity_error_tags"] or p["new_zero_intent_or_help"]]
    summary = {"status": "complete_270_calibrated_AI_ratings", "metrics_sums": metrics,
        "paired_repetition": dict(Counter(f"{p['control']['repetition']}->{p['candidate']['repetition']}" for p in pairs)),
        "aggregate_nonregression_failures": aggregate_failures, "per_id_veto_ids": vetoes,
        "adoption_approved": False, "regression_ids": [p["id"] for p in regressions],
        "improvement_ids": [p["id"] for p in improvements], "historical_control_difference_ids": [p["id"] for p in differences],
        "rubric_change": False, "raw_review_preserved": True, "independent_blind_or_human": False,
        "no_new_generation_or_training": True, "scope": "fixed development diagnostic, not generalization"}
    for name, payload in [("ratings.json", ratings), ("paired-comparison.json", pairs), ("calibration-changes.json", changes),
                          ("historical-rating-differences.json", differences), ("summary.json", summary)]:
        r.save(OUT / name, payload)
    sources = [BASE / "final-freeze.json", BASE / "calibration-decisions.json", Path(__file__).resolve()]
    files = list(OUT.glob("*.json"))
    r.save(OUT / "freeze.json", {"sources_sha256": {p.relative_to(r.ROOT).as_posix(): r.digest(p) for p in sources},
        "files_sha256": {p.relative_to(r.ROOT).as_posix(): r.digest(p) for p in files},
        "status": summary["status"], "raw_review_freeze_sha256": r.digest(BASE / "final-freeze.json")})
    print(json.dumps({"metrics": metrics["all135"], "paired_repetition": summary["paired_repetition"],
                      "veto_ids": vetoes, "remaining_historical_differences": len(differences)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
