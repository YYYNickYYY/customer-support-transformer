# 流程前缀对照 v1：预登记

2026-10-06。本轮止于方法验收，不运行第五模型，不prepare实际实验，不训练/调参/改数据。

## 问题与干预范围

在相同长度的完整流程前缀后，问题的部分商品/整单范围是否落实为正确下一对象与全文任务？账单领域前缀不适用于移除商品/取消整单时，续写能否转回订单任务？

新刺激属于看过失败后的探索性诊断，不是独立确认性业务评估。前缀是人工提供的助手历史，不是模型自然生成。共用问候与登录步骤，第二步仅替换领域线索；不包含selected/items/entire/cancel/remove等答案，不断言已执行或已获取取消资格。

本实验使用“review/check信息”的完整句来配平token位置，**不精确复现旧Navigate to Billing or Invoices路线**。neutral也是登录流程背景，不是无前缀基线；订单领域前缀也可能不必要。只比较这些固定领域线索，不能证明长度遗忘、特定注意力机制、上线安全或总体理解。

## 固定刺激与规模

两问题使用旧实验同句remove表述；每对仅范围两个token不同。不添加动词因素。remove整单不如cancel自然，这是固定合成刺激的限制，不声称分布内业务代表性。每种领域有两组12token完整句（A:Review details; B:Check information），域内措辞变化不能只报告有利的一组。全部历史token来源单独保存，整段重编码一致，无尾随独立空白。不给强制下一动作stem、候选或评价标准。

2问题 × 3领域 × 2措辞=12主体续写；重放partial/neutral/A与partial/billing/A各一次，共14。不给候选评分或局部probe，避免把代理概率当业务能力。generation保留每步token、输入位置、EOS和top5；top5缺失不当作概率零，不用于推断未走到的路径。

第五best.pt、CPU2/seed42、原greedy、context512、历史+新回复预算320、no_repeat0、角色token屏蔽、EOS允许。全部前缀等token长度，同一措辞三领域只变一个token，两问题prompt等长。prepare时额外冻结真实环境/全部token/权重/依赖/14旧Excel；本轮不创建实际outputs目录。

## 语义判断（先全文判读，不自动关键词评分）

只把**新生成内容**当成功证据；前缀里已有order/billing不能计作下一对象正确。评审记录源ID、证据原文、理由及帮助/风险(v1.1)/实体/重复/通顺(v1.0)旁注，不产生总分或因素表。

- next_object：partial（指定商品或询问具体商品）、whole（整单操作）、other（明确账户/地址/发票等其它对象）、mixed、unavailable。purchase单独出现不足以确定范围；your/order单token不充当分类器。完整操作对象与登录账户等普通准备动作区分。
- next_action：matching/conflicting/mixed/unavailable。忽略问候和纯准备，找第一个完整、明确的任务推进动作；通用“提供详情”、登录、泛查信息仍unavailable，不能向后跳过明确错误动作挑有利句子。询问具体商品可匹配部分诉求；整单澄清须针对订单标识/取消意愿，而非问待删商品。
- full_task：matching/conflicting/mixed/unavailable。开头正确后来转任务必须分开记录；matching不是操作已完成或路径经业务验证。全文不完整/预算截断可unavailable，不替模型补全。
- recovery（仅billing条件）：explicit_return（明确指出账单路线不适用并正确转回订单）、implicit_return（没有明说纠错，但新生成下一实质动作及全文均正确回应问题）、continues_wrong_domain（实质延续账单/发票任务或其它错误业务）、mixed（转回与转错并存）、unavailable（无充分行动/仅声明或截断）。非billing记not_applicable。说“回到订单”但无正确推进不算成功；提供部分/整单任一关键词不自动算恢复。
- 每个标签需引用新生成原文；unavailable可空证据但理由不得空。recovery证据只能来自续写。重复控制除id外全部判读一致。AI评审知道协议，不称独立人工盲评。

## 预设汇总与停止规则

逐条件保留两问题的下一对象/动作、全文与恢复判断。主要问题：同一领域/措辞下**双方是否分别next_action与full_task均matching**；任一unavailable则该配对无法判断。报告两措辞一致/不一致，不用14条当总体准确率，不把控制计入分母。不事后修改问题、前缀或判据。

账单续写都转回：仅说明这些前缀下能维持任务；账单比中性更易转错且两措辞一致：支持领域流程线索干扰的局部观察，不证明纯因果机制；各领域都失败：无法把失败专归账单；措辞分歧：保留措辞依赖。样本仅一对，不估计统计显著性。

先验收后另授权执行prepare→generate→人工判读→freeze-review→summarize。prepare必须验证机器准入hash；损坏/冲突记录先拒绝再加载模型。同hash只补缺，旧记录不覆写。重复控制token/文本/停止精确相同且top5顺序相同、概率差<=1e-6；每阶段逐张量状态不变、梯度None、optimizer_updates=0及依赖hash一致，否则标invalid并停止，不换病例。

## 唯一刺激清单

```json
{
  "version": "workflow-prefix-control-v1",
  "status": "preregistered_method_only",
  "date": "2026-10-06",
  "common": "Thank you for your message. Let me address your question.\n\n1. Log in to your account on our website.\n",
  "questions": [
    {"id": "partial", "instruction": "Please help me remove the selected items."},
    {"id": "whole", "instruction": "Please help me remove the entire order."}
  ],
  "prefixes": [
    {"id": "neutral-a", "domain": "neutral", "wording": "a", "route": "2. Review the relevant details before choosing the next action."},
    {"id": "order-a", "domain": "order", "wording": "a", "route": "2. Review the order details before choosing the next action."},
    {"id": "billing-a", "domain": "billing", "wording": "a", "route": "2. Review the billing details before choosing the next action."},
    {"id": "neutral-b", "domain": "neutral", "wording": "b", "route": "2. Check the relevant information before proceeding with this request."},
    {"id": "order-b", "domain": "order", "wording": "b", "route": "2. Check the order information before proceeding with this request."},
    {"id": "billing-b", "domain": "billing", "wording": "b", "route": "2. Check the invoice information before proceeding with this request."}
  ],
  "controls": ["partial-neutral-a", "partial-billing-a"],
  "generation": {
    "checkpoint": "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537/best.pt",
    "checkpoint_sha256": "2a2e549f954d96325daf4cb5c286edae9f9441b2ea806bab0bcdd1734fa46479",
    "tokenizer": "artifacts/tokenizer/bitext-bpe-v1/tokenizer.json",
    "tokenizer_revision": "43b80bbb10695d80aa442fa8435eb3ca8c33111554bf1d076006dcf2b7d76ea7",
    "model_config": "configs/model.json",
    "device": "cpu", "cpu_threads": 2, "seed": 42, "context": 512,
    "total_reply_budget": 320, "no_repeat_ngram_size": 0,
    "decoding": "greedy", "eos_allowed": true,
    "blocked_tokens": ["<PAD>", "<BOS>", "<USER>", "<ASSISTANT>"]
  },
  "evidence_sha256": {
    "outputs/opening-scope-control-v1-20261006/completion-freeze.json": "35f11abccbe88f20cd413f5aedf5106e0f35b99c9fa432046ee5b56a095a9eea"
  }
}
```
