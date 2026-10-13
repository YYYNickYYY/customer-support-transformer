"""Persist explicit continuation judgments; keep tokenization confounds visible."""
from collections import Counter
from pathlib import Path
import json
import re

import run_decoding_comparison as r

ROOT = r.ROOT
OUT = ROOT / "outputs/reply-prefix-20261006-v1"
PRIOR = ROOT / "outputs/condition-use-20261006-v1"

# Evaluated after complete continuation reading; supplied opening alone never earns match.
JUDGMENTS = [
    ("partial-1-original-neutral", "mismatch", "whole_order_cancel",
     ["delete purchase number {{Order Number}}", "complete the cancellation process"],
     "续写删除整笔购买并完成取消，未选择待删商品；开头raints碎词。"),
    ("partial-1-original-task", "mismatch", "whole_order_cancel",
     ["To delete your purchase", "smooth cancellation process"],
     "强制前缀明确部分商品，但续写整笔购买编辑/取消，没有商品选择或澄清；不能把强制文字当正确生成。"),
    ("partial-2-original-neutral", "match", "specific_item_clarification",
     ["specific items you would like to delete"],
     "全文开头delete your order范围含混，但唯一实际协助是询问具体待删商品，无整单取消路径；按主任务澄清通过，措辞冲突和碎词单列，非充分帮助。"),
    ("partial-2-original-task", "mismatch", "whole_order_removal",
     ["To delete your order", "confirm the removal"],
     "续写定位整单并确认删除，没有逐商品选择；强制部分商品开场未落实。"),
    ("whole-1-original-neutral", "match", "whole_order_cancel",
     ["To cancel your purchase", "complete the cancellation process"],
     "新生成的正文提供取消路径，任务正确；raints开场及字段重复是质量缺陷。"),
    ("whole-1-original-task", "match", "whole_order_cancel",
     ["To cancel your purchase", "Initiate Cancellation"],
     "续写仍为整单取消，非仅复述强制开场；feedback重复及碎词保留。"),
    ("whole-2-original-neutral", "match", "whole_order_cancel",
     ["To cancel your purchase", "Initiate Cancellation"],
     "正文取消流程符合任务；碎词及字段重复单列。"),
    ("whole-2-original-task", "match", "whole_order_cancel",
     ["To cancel your purchase", "complete the cancellation process"],
     "续写提供取消路径，非仅强制开场主题。"),
    ("request-1-original-neutral", "match", "refund_request_clarification",
     ["requesting a refund", "more details about the situation"],
     "续写申请主题及相关情况澄清；很泛，任务通过为弱证据，不能当完整申请步骤；vailure碎词。"),
    ("request-1-original-task", "mismatch", "refund_eligibility_substitution",
     ["Our refund policy", "Here are some common situations", "generally eligible for a refund"],
     "强制submit开场后主体转资格清单，无申请路径；末尾如果有具体订单问题可提供编号为条件式泛帮助，并非主任务申请流程，不能借closing澄清免除替代。"),
    ("request-2-original-neutral", "match", "refund_request_clarification",
     ["process of obtaining a refund", "transaction or transaction numbers"],
     "退款申请主题及交易信息澄清，措辞生硬和transaction重复不等于任务变成进度。"),
    ("request-2-original-task", "match", "refund_request_channel",
     ["To initiate the refund", "reaching out to our customer support team"],
     "续写申请渠道有效，不靠强制文字通过；necessary information/guide明显冗余及rau碎词单列。"),
    ("policy-1-original-neutral", "match", "refund_eligibility_scenarios",
     ["cases in which you can request a refund", "Product/Service Defect", "refund policies may vary"],
     "续写给资格情形和政策差异，业务事实未核验；illustrious突兀开头、末尾截断不改任务判断。"),
    ("policy-1-original-task", "match", "refund_eligibility_scenarios",
     ["Product/Service Defect", "refund policies may vary"],
     "新生成情形说明符合资格，不只靠强制eligibility；rauances碎词、末尾截断保留。"),
    ("policy-2-original-neutral", "match", "refund_eligibility_scenarios",
     ["Here are some common scenarios", "refund policies may vary"],
     "续写主体回答资格，但开头孤立Hours}}字段及病句是重大外观缺陷；任务匹配不等于回复可采用。"),
    ("policy-2-original-task", "match", "refund_eligibility_scenarios",
     ["common situations where you may be eligible for a refund", "refund policies may vary"],
     "正文资格对应；rauances开头与客套截断单列。"),
]


def summarize_probes(probes):
    summaries = []
    for source in dict.fromkeys(p["source_original_id"] for p in probes):
        stages = [p for p in probes if p["source_original_id"] == source]
        base = next(p for p in stages if p["stage"] == 0)
        summaries.append({"source_original_id": source, "family": base["family"],
            "stages": [{"stage": p["stage"], "available": p["available"], "prefix_tokens": p["prefix_tokens"],
                "own_correct_margin": p["own_correct_margin"], "condition_separation": p["condition_separation"],
                "margin_change_from_zero": p["own_correct_margin"] - base["own_correct_margin"] if p["available"] else None,
                "separation_change_from_zero": p["condition_separation"] - base["condition_separation"] if p["available"] else None,
                "ends_in_numbered_list_marker": bool(p["available"] and p["prefix_text"] and re.search(r"(?:^|\n)\s*\d+\.$", p["prefix_text"]))}
                       for p in stages]})
    later = [s for row in summaries for s in row["stages"] if s["available"] and s["stage"] > 0]
    return {"sources": summaries, "registered_stage_records": len(probes),
            "available_stage_records": sum(p["available"] for p in probes),
            "available_later_stage_records": len(later),
            "later_separation_below_zero_stage": sum(s["separation_change_from_zero"] < 0 for s in later),
            "all_available_separation_positive": all(p["condition_separation"] > 0 for p in probes if p["available"]),
            "numbered_marker_stage_records": sum(s["ends_in_numbered_list_marker"] for s in later),
            "warning": "Stage records not independent; numbered markers are punctuation not semantic sentence boundaries; signed probe margin is not task correctness"}


def main():
    for name in ("freeze.json", "execution-freeze.json"):
        r.verify_hashes(r.read(OUT / name)["sha256"])
    r.verify_hashes(r.read(PRIOR / "assessment-freeze.json")["sha256"])
    rows = [json.loads(line) for line in (OUT / "replies.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {v["id"]: v for v in rows}
    prior = {v["id"]: v for v in r.read(PRIOR / "assessments.json")}
    assessments = []
    for row in rows[:8]:
        old = prior[row["original_id"]]
        assessments.append({"id": row["id"], "source_id": row["source_id"], "family": row["family"],
            "prefix_condition": "empty", "verdict": old["verdict"], "action_match": old["action_match"],
            "actual_action": old["actual_action"], "evidence": old["evidence"], "notes": old["notes"],
            "side_observations": old["side_observations"], "provenance": "verified exact replay; preserved prior full-text judgment"})
    for sid, verdict, action, evidence, note in JUDGMENTS:
        row = by_id[sid]
        assert all(q in row["continuation"] for q in evidence), sid
        assessments.append({"id": sid, "source_id": row["source_id"], "family": row["family"],
            "prefix_condition": row["prefix_condition"], "verdict": verdict, "action_match": verdict == "match",
            "actual_action": action, "evidence": evidence, "notes": note,
            "side_observations": ["开场续写碎词/残片；BPE末尾独立空白混杂，不用于干净任务前缀因果结论",
                "风险话术/重复/字段/通顺仅旁注，未写入历史评分表"],
            "provenance": "AI full continuation reading; supplied prefix alone excluded from evidence"})
    control = dict(next(a for a in assessments if a["id"] == rows[8]["id"]), id=rows[-1]["id"], prefix_condition="control")
    assessments.append(control)
    assert len(assessments) == len(by_id) == len({a["id"] for a in assessments}) == 25
    for item in assessments:
        assert all(e in by_id[item["id"]]["response"] for e in item["evidence"])
    metrics = {}
    for family in ("partial", "whole", "request", "policy"):
        metrics[family] = {}
        for condition in ("empty", "neutral", "task"):
            selected = [a for a in assessments if a["family"] == family and a["prefix_condition"] == condition]
            metrics[family][condition] = {"n": len(selected), "matched": sum(a["action_match"] for a in selected),
                "verdicts": dict(Counter(a["verdict"] for a in selected))}
    probes = summarize_probes(r.read(OUT / "trajectory-probes.json"))
    forced = [v for v in rows if v["prefix_condition"] != "empty"]
    summary = {"status": "completed_with_instrument_limitations", "metrics": metrics,
        "trajectory": probes, "stops": dict(Counter(v["stop_reason"] for v in rows)),
        "forced_prefix_records": len(forced), "forced_prefix_texts_end_space": all(v["prefix_text"].endswith(" ") for v in forced),
        "forced_prefix_causal_task_interpretation_valid": False,
        "limitation": "Frozen artificial prefixes end with independently encoded whitespace; fragment/orphan-field continuations confound semantic intervention. No silent retry.",
        "baseline_exact_replay": 8, "identical_neutral_control": True, "optimizer_updates": 0,
        "training_started": False, "reviewer": "context-aware AI full-text; not independent blind human",
        "interpretation": "Fixed-branch conditional separation diminishes vs empty-prefix but stays positive; not monotonic, not proof of task forgetting. Artificial-prefix inference is exploratory only."}
    r.save(OUT / "assessments.json", assessments)
    r.save(OUT / "assessment-summary.json", summary)
    paths = [OUT / name for name in ("assessments.json", "assessment-summary.json", "execution-freeze.json", "freeze.json")]
    paths += [Path(__file__), ROOT / "docs/reply-prefix-results-20261006.md"]
    r.save(OUT / "assessment-freeze.json", {"sha256": {p.relative_to(ROOT).as_posix(): r.digest(p) for p in paths}})
    print("Completed 25 full-text judgments; 31/32 available stage records; artificial-prefix limitation retained.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
