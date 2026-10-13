"""Read-only mechanism analysis; deliberately does not assign semantic scores."""

from collections import Counter
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("decoding_runner", Path(__file__).with_name("run_decoding_comparison.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def repeated_fourgrams(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(max(0, len(ids) - 3))]
    return len(grams) - len(set(grams))


def summarize():
    root, out = runner.ROOT, runner.OUT
    frozen = runner.read(out / "generation-freeze.json")
    runner.verify_hashes(frozen["sha256"])
    runner.verify_hashes(frozen["dependencies_sha256"])
    replies = {arm: [runner.read(p)["payload"] for p in sorted((out / "traces" / arm).glob("*.json"))]
               for arm in ("control", "candidate")}
    pairs = runner.read(out / "mechanism-pairs.json")
    scopes = {"all135": pairs, "original27": [p for p in pairs if p["cohort"] == "original27"],
              "new108": [p for p in pairs if p["cohort"] == "new108"]}
    scopes.update({"intent:" + i: [p for p in pairs if p["intent"] == i] for i in sorted({p["intent"] for p in pairs})})
    metrics = {name: {"n": len(rows), "changed_texts": sum(not p["same_text"] for p in rows),
                     "changed_choice_steps": sum(p["constraint_changed_steps"] for p in rows),
                     "control_cutoff": sum(p["control_stop"] != "eos" for p in rows),
                     "candidate_cutoff": sum(p["candidate_stop"] != "eos" for p in rows)}
               for name, rows in scopes.items()}
    literal = {arm: {"affected_replies": sum(repeated_fourgrams(r["generated_token_ids"]) > 0 for r in rows),
                     "excess_occurrences": sum(repeated_fourgrams(r["generated_token_ids"]) for r in rows),
                     "total_generated_tokens": sum(r["generated_tokens"] for r in rows),
                     "stop_counts": dict(Counter(r["stop_reason"] for r in rows))}
               for arm, rows in replies.items()}
    assert len(replies["control"]) == len(replies["candidate"]) == len(pairs) == 135
    assert literal["candidate"]["excess_occurrences"] == 0
    secondary = runner.read(root / "outputs/fifth-generation-diagnostic-20261006-v1.1/selection.json")
    by_id = {p["id"]: p for p in pairs}
    runner.save(out / "mechanism-analysis.json", {"scope_metrics": metrics, "literal_fourgrams": literal,
        "secondary14": [{"id": s["id"], "stratum": s["stratum"], **by_id[s["id"]]} for s in secondary],
        "interpretation": "Exact token ngrams are NOT the v1.0 semantic repetition score.",
        "semantic_scores": None, "entity_audit_all135": None})
    by_arm = {arm: {r["id"].rsplit(":", 1)[-1]: r for r in rows} for arm, rows in replies.items()}
    evidence = []
    for short_id, before, after, reason in [
        ("1710", "order number {{Order Number}} and click on it.", 'order number {{Order History" section.',
         "第二次引用已有订单标识时，正确占位符变为未闭合的另一字段；开头保留标识不能抵消后文破损。"),
        ("11494", "To delete your {{Account Category}} account", "To delete your {{Account Type}} account",
         "正常复用客户已给账户类别时，改成客户未给的另一占位符；没有业务证据证明两字段同义。")]:
        a, b = by_arm["control"][short_id], by_arm["candidate"][short_id]
        assert before in a["response"] and after in b["response"]
        evidence.append({"id": a["id"], "instruction": a["instruction"],
                         "control_quote": before, "candidate_quote": after,
                         "new_entity_regression": True, "reason": reason,
                         "reviewer": "AI focused full-text paired audit; arm labels known, not blind",
                         "rubric_contract": "preregistration-v1 entity guardrail", "date": "2026-10-06"})
    runner.save(out / "focused-entity-veto.json", {"status": "focused_regression_evidence_not_full_scoring",
        "evidence": evidence, "adoption_approved": False,
        "reason": "Observed new per-ID entity defects violate the preregistered veto; do not adopt raw size=4.",
        "uncompleted": ["270 full semantic ratings", "135-pair complete entity audit", "complete quality deltas"]})
    runner.verify_hashes(frozen["sha256"])
    paths = [out / "generation-freeze.json", out / "mechanism-analysis.json", out / "focused-entity-veto.json",
             root / "scripts/summarize_decoding_comparison.py", out / "control-parity.json"]
    runner.save(out / "analysis-freeze.json", {"sha256": {p.relative_to(root).as_posix(): runner.digest(p) for p in paths},
        "status": "mechanism_complete_full_scoring_pending", "source_generation_status": frozen["status"]})
    print("135 paired mechanism rows verified; candidate exact 4-gram repeats=0 (NOT semantic score).")
    print("Two focused entity regressions retained; full semantic/entity scoring remains pending.")


if __name__ == "__main__":
    summarize()
