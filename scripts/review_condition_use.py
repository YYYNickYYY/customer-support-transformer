"""Explicit full-text judgments and paired summaries for the frozen small diagnostic."""
from collections import Counter
from pathlib import Path
import json

import run_decoding_comparison as r
from my_project_1.generation_diagnostic import locate_quote
from my_project_1.tokenization import load_frozen

OUT = r.ROOT / "outputs/condition-use-20261006-v1"
# id, verdict, actual action, exact evidence, full-text rationale, side observations
JUDGMENTS = [
    ("partial-1-original", "mismatch", "whole_order_cancel",
     ["removing some products", '"Cancel Purchase"', "complete the cancellation process"],
     "开场识别部分商品，后续按整笔购买取消，无商品选择或商品澄清，不能因主题词正确通过。",
     ["Order History/Edit重复", "no longer wish to remove it反向开场措辞", "订单号字段保留"]),
    ("partial-1-paraphrase", "mismatch", "whole_order_cancel",
     ["removing some products", "To proceed with the cancellation", "complete the cancellation process"],
     "删除部分商品改为整单取消流程，未提供逐商品动作。",
     ["prompts or prompts重复", "占位字段重复", "订单号字段保留"]),
    ("partial-2-original", "ambiguous", "generic_order_edit_removal",
     ["deleting some items", '"Order History" or "Billing"', '"Edit" or "Edit"', "complete the removal process"],
     "没有明确整单取消，但仅打开订单编辑后笼统removal，无逐商品选择/操作或相关商品澄清，按预登记条件未能确认满足；不等同明确整单错任务。",
     ["Edit重复", "Billing作为订单修改导航可疑", "订单号字段保留"]),
    ("partial-2-paraphrase", "mismatch", "whole_order_cancel",
     ["removing items", "complete the cancellation process"],
     "开场部分删除，正文结束取消整笔购买，未说明选定商品。",
     ["prompts or prompts重复", "smooth and hassle-free话术未在本轮重新风险打分", "订单号字段保留"]),
    ("whole-1-original", "match", "whole_order_cancel",
     ["To cancel your purchase", "Confirm Cancellation"],
     "给出整单定位及取消确认流程，满足位置/操作查询。",
     ["Online Order Interaction字段重复", "satisfaction话术仅旁注", "订单号字段保留"]),
    ("whole-1-paraphrase", "match", "whole_order_cancel",
     ["To cancel your purchase", "Initiate Cancellation"],
     "仍给整单取消路径，未转成部分商品。",
     ["ask for further information重复", "订单号字段保留"]),
    ("whole-2-original", "match", "whole_order_cancel",
     ["To cancel your purchase", "Confirm the Cancellation"],
     "明确整单取消帮助，提供对应流程。",
     ["Online Order Interaction字段重复", "smooth cancellation话术仅旁注", "订单号字段保留"]),
    ("whole-2-paraphrase", "match", "whole_order_cancel",
     ["To cancel your purchase", "complete the cancellation process"],
     "等义求助仍正确整单取消。",
     ["Online Order Interaction字段重复", "订单号字段保留"]),
    ("request-1-original", "match", "refund_request_clarification",
     ["requesting a refund", "more details about the situation"],
     "申请主题及相关情况澄清，没有替换为进度或仅资格；澄清很泛，任务通过不等于提供充分步骤或具强区分证据。",
     ["smooth refund process话术仅旁注", "无具体实体要求", "I work together措辞生硬"]),
    ("request-1-paraphrase", "mismatch", "refund_status_clarification",
     ["checking the status of your refund", "relevant updates"],
     "apply退款申请被解释为已申请后的状态/更新，相关澄清围绕进度而非申请；句子不同不是正确任务变化。",
     ["most accurate information重复", "无具体实体要求"]),
    ("request-2-original", "match", "refund_request_clarification",
     ["reason for the refund and any relevant purchase or transaction numbers", "guide you through the necessary steps"],
     "申请退款的相关购买/原因澄清，允许尚未给完整操作路径。",
     ["smooth refund process话术仅旁注", "无具体实体要求"]),
    ("request-2-paraphrase", "match", "refund_request_clarification",
     ["order number, date, and any relevant documentation", "obtaining a refund"],
     "仍是申请退款相关信息澄清，没有转资格或进度。",
     ["smooth refund process话术仅旁注", "无具体实体要求"]),
    ("policy-1-original", "match", "refund_eligibility_scenarios",
     ["situations in which you can request a refund", "Here are some common scenarios", "refund policies may vary"],
     "按问题说明资格情形并提示具体政策差异；最后客套截断不否认前文已回应任务，业务事实未核实。",
     ["max_new_tokens截断客套", "receive a refund等措辞不在本轮重新风险评分", "无具体实体要求"]),
    ("policy-1-paraphrase", "match", "refund_eligibility_clarification",
     ["clarity on the situations in which you can request a refund", "more details about the refund you are referring to"],
     "资格主题下请求具体退款情况，按预登记允许相关澄清；未讲出资格条件，属于弱任务通过，不是充分帮助或独立理解证据。",
     ["most accurate information重复", "I work together措辞生硬", "无具体实体要求"]),
    ("policy-2-original", "match", "refund_eligibility_scenarios",
     ["situations in which you can request a refund", "Here are some common scenarios", "refund policies may vary"],
     "问资格得到情形说明，非申请步骤替代；末尾截断只影响完整性。",
     ["max_new_tokens截断客套", "resolving the issue措辞重复", "风险话术仅旁注", "无具体实体要求"]),
    ("policy-2-paraphrase", "match", "refund_eligibility_scenarios",
     ["checking the refund policy", "Here are some common scenarios", "refund policies may vary"],
     "改述仍回应资格情形，末尾截断不改变任务判断。",
     ["max_new_tokens截断客套", "风险话术仅旁注", "无具体实体要求"]),
    ("identical-control", "mismatch", "whole_order_cancel",
     ["removing some products", '"Cancel Purchase"', "complete the cancellation process"],
     "和首条输入/输出完全相同，缺陷判定一致；控制是重复性检验，不增加独立病例分母。",
     ["与首条相同旁注", "订单号字段保留"]),
]


def main():
    frozen = r.read(OUT / "freeze.json")
    r.verify_hashes(frozen["sha256"])
    r.verify_hashes(r.read(OUT / "generation-freeze.json")["sha256"])
    plan = r.read(OUT / "plan.json")
    rows = {v["id"]: v for v in [json.loads(line) for line in (OUT / "replies.jsonl").read_text(encoding="utf-8").splitlines()]}
    tokenizer, _ = load_frozen(r.ROOT / plan["generation"]["model_config"]["tokenizer_artifact"])
    assessments = []
    for sid, verdict, action, evidence, note, observations in JUDGMENTS:
        row = rows[sid]
        if any(quote not in row["response"] for quote in evidence):
            raise ValueError("Nonliteral evidence: " + sid)
        assessments.append({"id": sid, "family": row["family"], "variant": row["variant"],
            "verdict": verdict, "action_match": verdict == "match", "actual_action": action,
            "evidence": evidence, "evidence_token_spans": [locate_quote(tokenizer, row["generated_token_ids"], quote) for quote in evidence],
            "notes": note, "side_observations": observations,
            "reviewer": "AI full-text, context-aware, not independent blind human assessment"})
    assert len(assessments) == len(rows) == len({a["id"] for a in assessments}) == 17
    judged = {a["id"]: a for a in assessments}
    metrics = {}
    for family in ("partial", "whole", "request", "policy"):
        metrics[family] = {}
        for variant in ("original", "paraphrase"):
            chosen = [a for a in assessments if a["family"] == family and a["variant"] == variant]
            metrics[family][variant] = {"n": len(chosen), "matched": sum(a["action_match"] for a in chosen),
                                       "verdicts": dict(Counter(a["verdict"] for a in chosen))}
    paired = [{"pair": pair, "original_both_match": all(judged[s]["action_match"] for s in pair),
               "paraphrase_both_match": all(judged[s.replace("original", "paraphrase")]["action_match"] for s in pair)}
              for pair in plan["pairs"]]
    stable = [{"original_id": c["id"], "both_task_correct": judged[c["id"]]["action_match"] and
               judged[c["id"].replace("original", "paraphrase")]["action_match"]}
              for c in plan["cases"] if c["variant"] == "original"]
    losses = r.read(OUT / "target-nll.json")
    assert len(losses) == 8 and len({v["case_id"] for v in losses}) == 8
    summary = {"status": "completed_small_supported_task_diagnostic", "metrics": metrics,
        "paired_tasks": paired, "source_paraphrase_stability": stable,
        "nll_positive_differences": sum(v["swapped_minus_matched"] > 0 for v in losses),
        "nll_comparisons": len(losses), "nll_min_difference": min(v["swapped_minus_matched"] for v in losses),
        "nll_max_difference": max(v["swapped_minus_matched"] for v in losses),
        "target_response_tokens_lost": sum(v["lost_response_tokens"] for v in losses),
        "training_target_truncated_sources": sum(v["original_training_retained_response_tokens"] < v["response_tokens"] for v in losses),
        "identical_input_control": r.read(OUT / "generation-summary.json")["identical_input_control"],
        "scope": "8 train sources, not dev/generalization accuracy; generic clarifications are weak positive evidence",
        "training_started": False, "optimizer_updates": 0,
        "interpretation": "Conditional likelihood preference exists; task execution and paraphrase robustness remain weak. Not causal localization."}
    r.save(OUT / "assessments.json", assessments)
    r.save(OUT / "assessment-summary.json", summary)
    paths = [OUT / name for name in ("assessments.json", "assessment-summary.json", "generation-freeze.json", "freeze.json")]
    paths += [Path(__file__), r.ROOT / "docs/condition-use-results-20261006.md"]
    r.save(OUT / "assessment-freeze.json", {"sha256": {p.relative_to(r.ROOT).as_posix(): r.digest(p) for p in paths}})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
