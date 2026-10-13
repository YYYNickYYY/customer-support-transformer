"""Validate explicit full-text probe judgments; do not infer meaning from text differences."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("input_runner", Path(__file__).with_name("run_decoding_comparison.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
OUT = r.ROOT / "outputs/input-sensitivity-20261006-v1"


def main():
    frozen = r.read(OUT / "freeze.json")
    gen = r.read(OUT / "generation-freeze.json")
    r.verify_hashes(frozen["sha256"])
    r.verify_hashes(gen["sha256"])
    assessments = r.read(OUT / "assessments.json")
    replies = {v["id"]: v for v in [json.loads(line) for line in (OUT / "replies.jsonl").read_text(encoding="utf-8").splitlines()]}
    judged = {v["id"]: v for v in assessments}
    if len(assessments) != len(judged) or set(judged) != set(replies) or len(judged) != 22:
        raise ValueError("Incomplete/duplicate judgments")
    for sid, j in judged.items():
        row = replies[sid]
        if type(j["action_match"]) is not bool or not j["notes"] or not j["evidence"]:
            raise ValueError("Missing semantic judgment")
        if any(q not in row["response"] for q in j["evidence"]):
            raise ValueError("Evidence not verbatim")
        for field, applies in (("negation_respected", row.get("negation_required", False)),
                               ("required_id_copied", "required_id" in row)):
            if (applies and type(j[field]) is not bool) or (not applies and j[field] is not None):
                raise ValueError("Invalid missing/nonapplicable field")
        if row.get("required_id") and j["required_id_copied"]:
            if row["required_id"] not in row["response"]:
                raise ValueError("False copy success")
    groups = []
    for group in frozen["plan"]["groups"]:
        cases = group["cases"]
        a, b = replies[cases[0]["id"]], replies[cases[1]["id"]]
        trace_a = r.read(OUT / "traces" / (a["id"] + ".json"))["payload"]
        trace_b = r.read(OUT / "traces" / (b["id"] + ".json"))["payload"]
        exact = a["generated_token_ids"] == b["generated_token_ids"] and a["response"] == b["response"] and a["stop_reason"] == b["stop_reason"]
        passed = exact if group["factor"] == "identical_input_control" else all(
            judged[c["id"]]["action_match"] and judged[c["id"]]["negation_respected"] is not False and
            judged[c["id"]]["required_id_copied"] is not False for c in cases)
        groups.append({"group": group["id"], "factor": group["factor"], "passed": passed,
            "cases": [c["id"] for c in cases], "first_pair_exact_output": exact,
            "first_pair_first_step_top5_differs": trace_a["steps"][0]["top5"] != trace_b["steps"][0]["top5"],
            "criterion": group["accept"]})
    summary = {"status": "completed_AI_full_text_diagnostic", "case_count": 22, "group_count": 9,
        "groups": groups, "negation_probes": {"n": 2, "respected": sum(j["negation_respected"] is True for j in assessments)},
        "explicit_copy_probes": {"n": 4, "correct": sum(j["required_id_copied"] is True for j in assessments)},
        "control_reproducible": next(g["passed"] for g in groups if g["group"] == "G09"),
        "scope": "synthetic mechanism probes, not generalization estimate; no aggregate weighted quality score",
        "reviewer": "AI full-text assessment against pre-frozen criteria; not independent human review",
        "training_started": False, "optimizer_updates": 0, "original_greedy": True,
        "interpretation": "Some broad task routing works; negation, copying and refund paraphrase fidelity fail. Text differences alone do not establish understanding."}
    r.save(OUT / "assessment-summary.json", summary)
    paths = [OUT / "assessment-summary.json", OUT / "assessments.json", OUT / "generation-freeze.json",
             OUT / "freeze.json", Path(__file__).resolve()]
    r.save(OUT / "assessment-freeze.json", {"sha256": {p.relative_to(r.ROOT).as_posix(): r.digest(p) for p in paths}})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
