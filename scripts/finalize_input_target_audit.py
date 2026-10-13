"""Persist explicit full-text AI judgments; never infer semantic scores by regex."""
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from audit_input_target import OUT, ROOT, create, digest, read, verify

# Suffix, task judgment, literal target evidence, explanation after reading full pairs.
DECISIONS = [
    ("101", "aligned", "Initiate the Cancellation Process", "无法取消的求助得到取消路径；占位符保留。"),
    ("10264", "aligned", "registering for a {{Account Type}} account", "否定拥有账户，而非拒绝开户；目标协助开户。"),
    ("104", "ambiguous", "Canceling the purchase could be an option", "输入只能确定无法负担，未明确取消；目标把取消作为选项，非无条件执行，但该层不能当显式整单取消正例。"),
    ("10927", "aligned", "account cancellation or deactivation", "不再使用并明确请求注销，目标注销；不是否定注销。另有feel占位符措辞缺陷，不扩展评分范围。"),
    ("12242", "aligned", "place an order from {{Delivery City}}", "目标回应城市配送/下单主题；全国配送断言未经业务核验，主题一致不等于事实获证。"),
    ("1247", "aligned", "items you would like to swap", "无法换商品的操作困难得到具体商品澄清，未反转成拒绝更换。"),
    ("1256", "aligned", "specific items you would like to delete", "部分商品删除，目标索取具体商品，不是整单取消。"),
    ("1298", "aligned", "specific items you would like to remove", "部分删除的必要澄清；执行保证另属风险，当前不代评分。"),
    ("1469", "aligned", '"Remove" or "Delete" option next to each item', "提供逐商品删除步骤，不改为整单取消；不验证实际网站具备按钮。"),
    ("14957", "aligned", "Locate the invoice with the number {{Invoice Number}}", "下载步骤对应诉求，但85632被字段化，非具体编号复制监督；没有显式要求复述。"),
    ("15094", "aligned", "invoice labeled with the number #{{Invoice Number}}", "不知道如何下载不是拒绝下载；37777被字段化。"),
    ("15460", "aligned", "specific invoice with the number {{Invoice Number}}", "下载路径正确；12588未保留，以通用字段替代。"),
    ("1568", "aligned", "specific details of the products you would like to be removed", "请求具体商品信息，范围是部分商品。"),
    ("15737", "aligned", "sacred number {{Invoice Number}}", "主题是取发票，但85632字段化；目标宣称取回而非实际工具结果，主题一致不等于可执行或有帮助。"),
    ("15782", "aligned", "invoice with number #{{Invoice Number}}", "发票获取主题对应；85632字段化且只承诺取回，不证明工具执行。"),
    ("1584", "aligned", "specific details of the products you wish to remove", "澄清待删除的具体商品，无整单取消。"),
    ("1607", "aligned", "specific item you would like to remove", "单商品删除的相关澄清。"),
    ("16160", "aligned", "reason for the reimbursement request and any relevant order or payment information", "退款申请的相关必要澄清，不是仅列资格。"),
    ("16228", "aligned", "timeframe for receiving a refund", "how long询问时长，不是申请步骤；目标时长解释与联系核查对应，候选误归不能判目标错。"),
    ("16432", "aligned", "guide you through the rebate process", "rebate申请转官方客服办理；具体结果承诺不在本轮评分范围。"),
    ("16597", "aligned", "First, gather any relevant documentation", "实际申请流程：准备凭证、联系支持、网站入口，非仅资格。"),
    ("16831", "aligned", "purchase confirmation email for specific instructions on how to request a refund", "给出申请渠道及订单号/原因，尊重不知道如何申请而非否定申请。"),
    ("16867", "aligned", "details of your purchase, such as the order number", "申请退款时索取相关购买信息，是可接受澄清。"),
    ("1698", "aligned", "specific details of the items you would like to remove", "删除多个商品的澄清，不是撤销整单。"),
    ("1709", "aligned", "specific items you would like to remove", "部分商品删除相关澄清；调整结果承诺另属风险。"),
    ("19647", "aligned", '"Add to Cart"', "不知道如何下单，目标给下单路径，不是拒绝下单。"),
    ("21382", "aligned", "more details about the issues you're facing", "无法注册的故障求助，目标请求故障信息。"),
    ("23382", "aligned", '"Add a New Address"', "不知道如何设置地址，目标设置步骤。"),
    ("26452", "aligned", "check the status of your reimbursement", "进度查询目标，不是申请或资格替代。"),
    ("26614", "aligned", "inquire about the status of your refund", "询问更新实际为进度任务；how候选不能当申请步骤。"),
    ("26713", "aligned", '"Refund Status"', "输入track，目标状态查询；dont缺撇号不命中宽否定，提示正则漏检，不改变冻结规则。"),
    ("378", "aligned", "option or link to cancel the order", "明确整单取消，目标取消路径。"),
    ("4224", "aligned", "invoice with number #{{Invoice Number}}", "查看发票主题和相关澄清，37777字段化。"),
    ("435", "aligned", "Initiate Cancellation", "明确取消得到取消流程。"),
    ("443", "aligned", "To cancel your purchase", "明确取消得到取消流程。"),
    ("460", "aligned", "Initiate the Cancellation", "明确取消路径及订单字段保留。"),
    ("4787", "aligned", "invoice with the number {{Invoice Number}}", "查看发票主题对应；00108前导零编号未保留，只承诺定位发票。"),
    ("4822", "aligned", '"Invoices" or "Billing"', "查看发票路径对应；12588字段化。"),
    ("528", "aligned", "guide you through the cancellation process", "不要这笔购买自然支持取消，不等于不想取消；重复索取已提供字段另列局限。"),
    ("612", "aligned", "To cancel your order", "不知道如何取消，目标取消流程；dont漏检不证明没有否定。"),
    ("6365", "aligned", "cases in which you can request a reimbursement", "输入资格问题，列资格符合任务，不是申请步骤错配。"),
    ("6461", "aligned", "cases in which you can request a refund", "询问哪些情形可退，目标资格说明对应。"),
    ("6666", "aligned", "cases where you can request a refund", "in which cases请求资格，不能因request命中就要求申请步骤；残留保证不在本轮重新打分。"),
    ("6688", "aligned", "check the cases in which you can request a refund", "输入是资格而非申请步骤；目标资格对应。"),
    ("734", "aligned", "please feel free to ask your question", "输入只说有取消疑问，目标请具体说明属于相关澄清。"),
    ("824", "aligned", "Cancel the Purchase", "明确询问取消位置，目标取消路径。"),
    ("9514", "aligned", '"Contact Us"', "不知道如何联系人工，目标提供渠道，不是拒绝联系人工。"),
    ("969", "aligned", "Initiate the Cancellation", "明确取消得到取消步骤。"),
]


def main():
    verify()
    for name, expected in read(OUT / "scan-freeze.json")["files_sha256"].items():
        assert digest(OUT / name) == expected
    rows = read(OUT / "samples.json")
    assert len(rows) == len(DECISIONS) == 48
    reviews = []
    for row, (suffix, judgment, evidence, note) in zip(rows, DECISIONS):
        assert row["id"].rsplit(":", 1)[-1] == suffix
        assert evidence in row["response"]
        ids = row["strata"]
        reviews.append({"id": row["id"], "strata": ids, "task_alignment": judgment,
                        "evidence": evidence, "reason": note,
                        "identifier_copy_support": "placeholder_substitution" if "identifiers" in ids else "not_applicable",
                        "reviewer": "AI full-text review, context-aware, not independent blind human review"})
    strata = {}
    for key, selected in read(OUT / "scan.json")["selected_ids"].items():
        subset = [r for r in reviews if r["id"] in selected]
        strata[key] = {"selected": len(selected), "reviewed": len(subset),
                       "pending": len(selected) - len(subset),
                       "task_alignment": dict(Counter(r["task_alignment"] for r in subset))}
    create(OUT / "reviews.json", reviews)
    create(OUT / "review-summary.json", {
        "unique_reviewed": len(reviews), "strata": strata,
        "interpretation": "Task alignment is not risk/helpfulness/copy correctness or population defect rate",
        "partial_items": {"reviewed": 8, "specific_item_clarification": 7, "item_level_steps": 1, "whole_order_substitution": 0},
        "refund_candidate_actual_tasks": {"request": 3, "timeframe": 1, "status": 2, "eligibility": 2},
        "direct_rejection_examples_reviewed": 0, "explicit_copy_requests_reviewed": 0,
        "sources_modified": False, "training": False})
    create(OUT / "review-freeze.json", {
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_sha256": {name: digest(OUT / name) for name in ("reviews.json", "review-summary.json")},
        "review_script_sha256": digest(Path(__file__)),
        "report_sha256": digest(ROOT / "docs/train-input-target-audit-20261006.md")})
    print("48 fixed pairs reviewed with literal evidence; no semantic auto-scoring or training.")


if __name__ == "__main__":
    main()
