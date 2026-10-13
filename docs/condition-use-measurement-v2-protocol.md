# 条件利用测量 v2：预登记协议

日期：2026-10-06。仅设计与方法验收；本轮不加载第五模型做新推理，不训练、不调参、不改监督话术、不读 sealed_test。协议在方法单测前写定；只用 tokenizer 检查过候选长度，未用模型输出挑选刺激。

## 1. 唯一问题与可以被否证的观察

**同一完整中性回复历史下，将问题从任务 a 换成任务 b，候选行动句的相对条件概率是否随之变化，实际续写是否落实对应任务？**

沿用先前已核查正确监督的两种区别：部分商品/整单取消、申请退款/退款资格。复用已冻结的 8 个原始问题，排名同位配成 4 对，不根据 v1 输出重新选样，不增加新业务、编号复制或反向否定。它们是已看过的训练来源诊断，不是独立 dev 测试，也不汇成总体理解率。

v2 明确收缩解释：测量候选行动句的**条件敏感性**，不声称测出“真正的任务偏好”，不声称共享人工历史证明自生成历史导致遗忘。该问题允许得到“分数变了但行动没有落实”或“候选表达之间不一致”。不能以任何一种情况直接决定训练因素。

## 2. 刺激、边界与下一行动契约

机器可读的唯一刺激清单在本文件末尾 JSON 块；测试从此处读取，不另建 plan、工作簿或重复清单。

共享历史只有 H0 空、H1 一句、H2 两句。非空历史均语义完整、中性、不承诺处理成功、不指向任一任务，末尾是句号，不保留独立空白 token。历史在同一配对两个问题下 token 完全相同。每个历史都尚未执行任何业务行动，因此下一步的诊断对象仍是回应客户当前请求；不是后半段已完成任务后的再次强塞行动。

每类 3 对候选全部用相近结构的可执行“查找相关选项/步骤/条件”指引，不拿索取信息与宣告操作作对比。候选不假设系统有执行权限，不规定未提供的退款政策，也不承诺订单能成功取消。查找选项只是低强度帮助，不等于实际业务解决。

- 部分：应指向选定商品的移除/逐商品处理，不将整单取消当替代。
- 整单：应指向整单取消，不仅说明删除部分商品。
- 申请：应指向发起退款的步骤/渠道/相关必要资料，不只解释资格。
- 资格：应指向退款条件或定位正式政策，不把发起申请本身当资格判断。

同一类三组行动说法逐组保存，不只保留最有利的一组。固定 token 长度差上限 2，不保证等长或等频；词频、表达与问题位置差异仍是解释限制。此门槛只排除明显不匹配，不证明构念有效。

### 编码与句末规则

使用冻结 byte-level BPE 与训练模板：分别编码 instruction 和 response，插入控制 token。分支字符串包含一个**前导**普通空格，历史无尾随空格；前导空格随行动首词编码，不提前强制生成一个独立空白历史。

所有 H+C 必须满足整体 response 编码 = encode(H)+encode(C)，且精确 decode；不合则停止，不事后剥离/补空格。采用人工声明的 H1/H2 完整句，不调用旧的标点截点算法，不使用 32/64/96 token 阈值。列表编号、小数、缩写都不产生自动“句末”。将来新增历史必须重审新版本，不能按输出移动截点。

模型分支输入 [BOS, USER, instruction, ASSISTANT, H, C] 错开一位，H/问题标签 -100，只对 C 的每个 token 计 NLL，**不额外计 EOS**（这是下一行动句，不是完整回复结束）。标签首个 C token 在最后历史 token 或 ASSISTANT 位置预测；不得把候选首 token 放进自身输入。所有输入完整容纳 512，不截断，不借 padding 改位置。不允许控制 token 出现在自然字段。

## 3. 数值指标：明确算的是什么

对配对问题 q_a/q_b、共享历史 h、候选 c_a/c_b：记录逐 token NLL、总 NLL、均值 NLL、token 数和输入长度。记 L 为均值 NLL，

M(q,h) = L(c_b | q,h) - L(c_a | q,h)

D(h) = M(q_a,h) - M(q_b,h)。D>0 表示这对指定句子的相对分数向与问题匹配方向变化；它不要求 M(q_a)>0 且 M(q_b)<0，也不是正确率或业务概率。总 NLL 同样计算一个 D_total，完整并列，不把二者任选成有利指标。

每对/历史的三组说法：若 3 个 D_mean 与 3 个 D_total 均 > 1e-6，记数值一致正向；均 < -1e-6 记一致反向；所有绝对值 <=1e-6 记数值持平；其他为说法/归一化不一致。1e-6 只是报告浮点近零，不是显著性或能力门槛。任何记录缺失则记 unavailable，不能补零或缩分母。

只在同一问题、同一历史、同一候选间构造预定对照。问题长度不同，绝对位置也可能不同：保存每边 prompt 与历史长度，不补造 padding 做“语义因果”声明。H0/H1/H2 的 D 差异仅是不同固定历史下的数值观察，不能叫遗忘速率、时间轨迹或历史长度因果效应。不能把 3 组说法、3 历史当独立样本做显著性检验。

## 4. 实际续写与语义判定（执行阶段才评）

同一 8 问题分别在 H0/H1/H2 后 greedy 续写，共 24 条；另重放首对两个问题 H1，各 1 条。总 26 条，其中 2 条控制不加入独立样本分母。生成只给问题与共享历史，不给候选、intent、参考目标或评分标准。320 为历史+新回复 token 总预算，EOS 开启，角色 token 屏蔽，重复约束为 0；固定第五 checkpoint、CPU2、seed42、context512，不使用旧 configs/generation.json 猜 checkpoint。

每条保留逐步 token、停止原因、完整 response 与 continuation。语义评审仅看问题/全文，先判读再查看数值分数，防止数值符号引导判断；评审仍为有上下文的 AI 辅助，不称独立盲评。

- 第一个实质行动：跳过问候/泛服务意愿，从第一个完整任务行动或必要澄清开始。matching / conflicting / mixed / unavailable 四选一。
- matching：明确落实本问题的行动方向，或提出该方向特有的必要澄清。泛“给我订单号/告诉我更多”不足以区分两任务，记 unavailable，而非自动通过。
- conflicting：明确转到相反任务并代替所求行动。两方向并列且不能确定主动作记 mixed。若合法资格提醒服务于申请流程，不能仅因提及资格就判冲突。
- 无可辨行动、句子残缺、开头生成未完而上限结束、只有无依据执行承诺：unavailable 并说明原因。不要把 EOS 正常停止自动当通过。
- 另判全文是否保持主任务（matching/conflicting/mixed/unavailable），以及帮助、重复、实体变化、无依据承诺与通顺的文字旁注。不生成新的总分，不回写历史因素表。

数值一致且实际下一行动/全文方向匹配，才称该局部对照支持“条件差异落实”；分数一致但生成失败是概率—生成差距；数值不一致则候选依赖；生成正确而分数不一致则测量代理受限。均不得升级为总体理解能力。第一行动与全文结论分开，防止开头正确掩盖后续转向。

### 预设语义边界示例（AI编写，不是独立人工金标准）

| 问题 | 下一行动片段 | 预设判读及理由 |
| --- | --- | --- |
| 删除部分商品 | Which items would you like to remove? | matching：询问具体待移除商品，不转整单 |
| 删除部分商品 | Please check your order options for cancelling the entire order. | conflicting：用整单取消替代部分移除 |
| 整单取消 | Please provide your order number. | unavailable：通用资料请求，无法据此区分部分/整单，不判错误 |
| 申请退款 | Please check the seller's instructions for submitting a refund request. | matching：定位申请步骤；帮助有限，但无执行承诺 |
| 申请退款 | Please check the seller's conditions for qualifying for a refund. | 第一行动方向为policy；仅此一句且无申请路径时全文conflicting；若后续给有效申请路径，则单列资格前置提醒，不自动判全文失败 |
| 退款资格 | Please check the seller's conditions for qualifying for a refund. | matching：定位条件，不虚构具体资格政策 |
| 退款资格 | Your refund has been approved. | unavailable并标无依据执行承诺：没有解释条件，不自动当匹配 |
| 任一任务 | You can remove selected items or cancel the entire order. | mixed：未根据明确诉求确定主行动 |

以上例子固定语义边界，但不由字符串匹配脚本代替全文评审。候选“检查条件”本身可合法服务于申请流程，因此实际续写要结合后文；候选分数只表示方向兼容性，不是一对唯一业务正确答案。

## 5. 方法验收门槛及后续执行准入

本轮仅做本地刺激、tokenizer、合成因果 oracle 与函数测试。通过准入须：来源/哈希一致，问题未替换，三个共享历史边界正确、全部候选自然字段无控制 token、每组长度差<=2、全部 144 分支可完整容纳、标签/oracle/模式恢复/无梯度通过、缺失/近零/说法冲突分类通过。

机器测试只验收数值和输入契约，**不能验收真实模型的业务构念**。当前候选与历史由本轮 AI 做内容审查，没有独立语义金标准。允许结论上限为“有限探索性诊断方案”；不能宣称模型任务选择测量已获完整验证。

若工程检查失败，本轮到此停止，不加载第五模型。若工程检查通过，也不自动运行第五：先另行确认执行，在推理前冻结本协议/测试/实现依赖、验收结论、实际运行环境和全部输入 token；若工程实现尚未覆盖记录/恢复/控制契约，先补执行器单测，再冻结后运行。无损恢复只能补同哈希缺项；冲突或损坏停止，不覆盖。

未来分支 4对×3历史×3说法×2问题×2候选=144 条评分 forward，加首对 H1 第一说法四条同输入重放=148；分支重放的逐token NLL应在CPU同环境下 abs<=1e-6、token数/输入长度精确一致。26次续写控制的 token/正文/停止须精确一致；任一控制失败整批失效，不挑其它结果。保存 unavailable、无截断、权重前后哈希、梯度 None、optimizer_updates=0 和未启用数据变更。模型运行的副作用与真实语义交叉判断尚未发生，本轮不伪造这些结果。

## 唯一机器刺激清单

```json
{
  "version": "condition-use-measurement-v2",
  "date": "2026-10-06",
  "status": "preregistered_method_only",
  "training": false,
  "fifth_inference": false,
  "tokenizer": "artifacts/tokenizer/bitext-bpe-v1/tokenizer.json",
  "config": "configs/model.json",
  "dependencies": {
    "src/my_project_1/tokenization.py": "9af3671c45504dfba09be278f1111a97324eb92742a5907cbac55a24e4383546",
    "outputs/condition-use-20261006-v1/plan.json": "6dc8997d740d5abc143110f6f2e26110fa99a7cfda389b6e79a2d5333259e371",
    "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537/best.pt": "2a2e549f954d96325daf4cb5c286edae9f9441b2ea806bab0bcdd1734fa46479",
    "artifacts/tokenizer/bitext-bpe-v1/config.json": "ac42fc8423c12ad1bca23370a9e55ae89aef89786c638f051160fd97cbdf2baf",
    "configs/model.json": "1c0afa156f7afb403cfde51b1b01d12b48775f8d5a76a45719f3a102d0e870d0",
    "src/my_project_1/reply_prefix.py": "db77bcf126ab0e1600fac57df4db33f018118ea27712d9eb526ab9eff1805213",
    "configs/training-fifth-final.json": "f151956dae46208d6a4fc15e4a5ef7ce65bd8d76505545b7eb1a0696a90b5dfc",
    "artifacts/tokenizer/bitext-bpe-v1/tokenizer.json": "43b80bbb10695d80aa442fa8435eb3ca8c33111554bf1d076006dcf2b7d76ea7",
    "data/derived/refund-execution-v3.1/train.jsonl": "4d1ed247e99b7b9aa5914be35ea9e6bcb8d4a8f70f1188808cda5b8bd07fbe73",
    "docs/reply-prefix-measurement-acceptance-20261006.md": "ddf0501bfd33e538cae1216b5df9c2f36c715b51668299f8e1540b720c3694b1"
  },
  "prefixes": [
    "",
    "Thank you for your message.",
    "Thank you for your message. Let me address your question."
  ],
  "families": {
    "order": {
      "a": "partial",
      "b": "whole",
      "branches": [
        [
          " Please check your order options for removing selected items.",
          " Please check your order options for cancelling the entire order."
        ],
        [
          " Look for information on removing selected items from your order.",
          " Look for information on cancelling the entire order."
        ],
        [
          " Check whether the order page offers removal of selected items.",
          " Check whether the order page offers cancellation of the entire order."
        ]
      ]
    },
    "refund": {
      "a": "request",
      "b": "policy",
      "branches": [
        [
          " Please check the seller's instructions for submitting a refund request.",
          " Please check the seller's conditions for qualifying for a refund."
        ],
        [
          " Look for the seller's steps for requesting a refund.",
          " Look for the seller's rules for refund eligibility."
        ],
        [
          " Check where the seller explains how to submit a refund request.",
          " Check where the seller explains which purchases qualify for a refund."
        ]
      ]
    }
  },
  "cases": [
    {
      "id": "order-1",
      "family": "order",
      "source_rows": [
        1469,
        824
      ],
      "questions": [
        "how can I delete some products from purchase {{Order Number}} ?",
        "where do I cancel purchase {{Order Number}}?"
      ]
    },
    {
      "id": "order-2",
      "family": "order",
      "source_rows": [
        1256,
        969
      ],
      "questions": [
        "want help to delete some items from order {{Order Number}}",
        "help me to cancel purchase {{Order Number}}"
      ]
    },
    {
      "id": "refund-1",
      "family": "refund",
      "source_rows": [
        16867,
        6461
      ],
      "questions": [
        "i need help to request a refund",
        "help checking in what cases can i request to be refunded"
      ]
    },
    {
      "id": "refund-2",
      "family": "refund",
      "source_rows": [
        16597,
        6688
      ],
      "questions": [
        "I don't know how I can obtain a refund",
        "assistance to check in which cases can I request a refund"
      ]
    }
  ],
  "candidate_length_difference_max": 2,
  "expected": {
    "pairs": 4,
    "questions": 8,
    "prefixes": 3,
    "wordings": 3,
    "branch_scores": 144,
    "repeat_branch_scores": 4,
    "continuations": 24,
    "repeat_continuations": 2
  },
  "future_generation": {
    "checkpoint": "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537/best.pt",
    "device": "cpu",
    "cpu_threads": 2,
    "seed": 42,
    "max_length": 512,
    "total_reply_budget": 320,
    "no_repeat_ngram_size": 0,
    "decoding": "greedy",
    "blocked_tokens": [
      "<PAD>",
      "<BOS>",
      "<USER>",
      "<ASSISTANT>"
    ],
    "eos_allowed": true
  },
  "interpretation": "candidate_sentence_condition_sensitivity_not_business_accuracy_or_forgetting"
}
```

