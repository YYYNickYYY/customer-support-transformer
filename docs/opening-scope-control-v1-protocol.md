# 开场动作范围选择：受控对照设计 v1

2026-10-06。状态：**仅设计与tokenizer预检；执行器未实现、方法单测未验收、未做新模型推理。** 不训练、不调学习率/轮数/模型，不改目标或活动生成配置，不读取sealed_test。

## 明确问题与来源

上一轮只读trace观察到：同一部分商品问题，在H1后能询问specific items，在H2后新增第3/4 token形成your order。该观察没有分离历史内容、长度及自生成措辞，本实验不能将其叫作已证实的遗忘。

本次只问：**把其它输入固定，在相同开场动词之后，将问题的范围从部分商品改成整单，范围候选分数与实际开场对象是否随之变化？这种局部现象是否依赖delete/remove动词匹配？**

这是看过既有失败后提出的新假设，下面的新刺激在看到其输出前固定，属于探索性机制诊断，不是独立确认性评估。不是复现旧原始问题的准确率，也不是新客服业务测评。

## 1. 固定小型因子矩阵

所有条件使用同一段H2完整中性历史：Thank you for your message. Let me address your question.（12 token）。不再改变历史长度，也不比较H1/H2来解释遗忘。

问题只有四句，交叉问题动词delete/remove与范围selected items/entire order；同一个问题动词内只换两个范围token。无反向否定、编号复制或新退款业务。范围语义由实验定义为移除选定商品/整单，并不代表实际商户支持删除订单。

| 问题动词 | 部分商品 | 整单 |
| --- | --- | --- |
| delete | Please help me delete the selected items. | Please help me delete the entire order. |
| remove | Please help me remove the selected items. | Please help me remove the entire order. |

问题均8 token，范围差异位于相同位置。整单表述在真实客服中可能更常用cancel，因此这是刻意配平的合成输入；不用它单独证明分布内能力。delete/remove在这里用于操作对象差异，不声称它们与cancel完全等义。

每个问题有三个输出条件：

- free：仅给完整H2，模型自己决定如何开场，作为生态参照。
- delete：给H2后接固定“ To delete”，随后自由续写。
- remove：给H2后接固定“ To remove”，随后自由续写。

主要范围对照：同一问题动词、同一固定开场动词下，比较部分/整单问题。次要动词匹配对照：同一问题、相同长度开场下，比较delete/remove。两个问题动词都报告，不能只保留同词复述最有利的一组。

**强制开场不是部署建议。** 它是窄范围干预，可能改变整个后续分布；free与强制开场的长度/自然性不同，二者只作为参照，不能将差异称为纯开场语义因果效应。

## 2. 改测完整范围短语，不把your当整单标签

“your”后仍可能是your selected items；“order”后也可能是order number。不能只看单token就判动作对象。固定开场之后用两组完整短语：

| 说法 | 部分候选 | 整单候选 | token数 |
| --- | --- | --- | --- |
| explicit | the selected items | the entire order | 3 / 3 |
| alternative | some items | your order | 2 / 2 |

实际字符串均含前导空格。每对等长、在同一位置评分，保留逐token与总/均NLL，问题/历史/强制stem不计loss，不计EOS、不截断。两个说法各自分析，不把3token与2token的总分跨说法直接相减。

explicit复用了问题范围短语，可能测到复制倾向；alternative提供不同措辞，但不是排除复制/词频影响的充分验证。要由两组是否一致及实际续写共同约束解释，不添加第三组直到结果漂亮。

完整范围短语仍只是语言兼容性候选，不是完整业务行动、订单可取消资格或真实执行结果。其它正确措辞不在两个候选中，不因此判错误。

## 3. 同一分叉上下文的下一token探针

为避免拿不同自生成位置的概率直接比较，预先固定两个局部上下文：

- after-the：H2 + stem + “ the”；记录Ġselected/Ġentire。
- after-verb：H2 + stem；记录Ġsome/Ġyour。

在每个局部上下文中，配对两问题除范围token外完全相同，输入长度/目标预测位置相同。另记录完整top5、EOS概率、两个预指定token ID及概率；无论是否出现在top5都直接读取其概率，禁止把top5缺失补为0。

探针采用与greedy相同角色屏蔽分布；EOS不屏蔽。不改变采样设置。first-token oracle须验证分支未来token不影响该位置；不从完整候选分数倒推被屏蔽分布。

定义R(q)=log p(partial token)-log p(whole token)，分别报告两问题的R与ΔR=R(q_partial)-R(q_whole)。after-verb的some/your只是词汇竞争，不是范围分类器；after-the的selected/entire也是预指定短语开头概率，不代表完整动作正确性。不跨两个探针比较绝对概率，也不把概率高称为业务有把握。

## 4. 预设指标与实际判读

完整候选的M(q)=NLL(whole|q)-NLL(partial|q)，D=M(q_partial)-M(q_whole)。每组同长，mean/total方向相同应成为算术检查；两组D分别保存。

- 两说法D均>1e-6：指定范围短语相对敏感性正向。
- 两说法D均<-1e-6：一致反向。
- 所有绝对值<=1e-6：数值近零。
- 其余：说法依赖；任何缺失则unavailable。

另报告每侧M，只有两说法在部分问题均M>1e-6、整单问题均M<-1e-6，才记“候选分别偏好自身方向”。D正向但未跨零不能叫双方选对。1e-6仅为数值近零，不是显著性/业务阈值；探针R也遵循同样解释边界，不将探针数量当独立样本数。

先完成14次续写并冻结语义判读，再计算候选与探针分数。语义判读者不先查看分数，但同一AI知道旧诊断和实验设计，不称独立盲评。

逐条判“第一个明确操作对象短语”与“全文任务落实”，分开记录：

- partial：明确操作选定/部分商品，或询问具体待移除商品。
- whole：明确操作整单对象。须结合操作动词与完整对象，不以出现order单词就判。
- mixed：两种对象并列、未根据诉求选定。
- unavailable：只有问候、登录、泛详情、残句或对象未完成；另记录其它任务，例如改地址，不能伪装成正确或泛未输出。

对象出现partial/whole不等于动作真实可行。按客户所求另判matching/conflicting/mixed/unavailable；全文有效澄清也可匹配，但无执行工具不判业务已完成。开头对象正确而后面转整单，须保留开头匹配/全文冲突。不要根据结果重新挑“第一个动作”。

“没有任何第一对象”的记录仍保留，并列停止原因/截断，不能无限向后寻找有利词组；读完整已保存回复后按上述规则判断首个完整明确操作对象，后面的任意复述不替代它。帮助、重复、实体、风险和通顺只做文字旁注，不产生新总分或改旧因素表。

## 5. 规模与控制

主要续写4问题×3开场=12；重放delete-partial的free/delete各1，共14次。候选仅在2个固定stem条件下评分，4问题×2stem×2说法×2候选=32；重复delete-partial/delete全部4分支，共36次。局部探针4问题×2stem×2探针=16；重复delete-partial/delete两探针，共18次独立forward。

两个续写控制token/正文/停止精确一致；候选及探针控制概率/NLL abs<=1e-6、token/长度一致。失败则整批无效，不挑别的组继续解释。记录权重/状态逐张量一致、梯度None、optimizer_updates=0及依赖前后哈希。

保持第五权重、冻结tokenizer、CPU2/seed42、greedy、context512、历史+stem+新回复总预算320、no_repeat0、角色屏蔽/EOS开启。这里的“固定开场”是明示实验干预，不暗改原greedy或强制范围候选。实际续写只能给问题+H2+指定stem，不泄露候选或评审标准。

## 6. 方法准入与停止规则

**本轮tokenizer预检已通过，完整方法尚未验收。** 四问题8token，两stem2token，两组候选3/3及2/2；H2+stem+候选与整段response重编码一致，没有强制独立末尾空白。最长候选完整序列为11prompt+12历史+2stem+3目标=28token。

特别区分：H2是完整句历史；stem刻意为未完成开场短语。它不满足旧v2“中性完整句历史”的语义用途，不能偷偷套用旧验收。新执行器必须把history与stem分字段、保留token来源，验收此窄范围cloze干预的契约；不能称人工未完成短语是自然对话历史。

执行前必须：

1. 问题/stem/候选/探针的token ID、位置、共用前缀及整段编码契约单测；禁止保留尾随独立空白。
2. 同一分叉位置用非均匀、历史/位置敏感oracle手算标签与log-odds；验证屏蔽与归一化、候选未来无泄露、无梯度、模式恢复和上下文拒截断。
3. 所有控制、create-only记录、同哈希补缺、冲突/损坏拒绝及先语义后分数门控测试。
4. 准入报告记录人工stem及复制/合成措辞限制；若失败停在方法层，不开始模型推理。
5. 通过后另行冻结本协议、验收、执行器/单测、全部输入token、真实环境、权重/tokenizer/旧证据，然后才执行。原v2文件不修改，旧trace只作问题来源。

若候选/局部探针随问题换向，但实际对象错：记录局部分数与生成差距。若D/ΔR正向但双方不分别偏好自身方向：记录偏置仍在，不能跳到“已理解但解码坏”。若结果强依赖delete/remove匹配或同词候选：记录词汇/表述依赖。若候选正确而探针或实际不一致：代理测量仍有限，不据此改模型。

任何结果都不定位特定注意力头，不证明历史长度导致遗忘，不自动改数据/学习率/轮数/模型。本实验小、非独立业务测试；不足以支持因果认知机制或总体能力声明。

## 唯一刺激清单（设计阶段，未执行）

```json
{
  "version": "opening-scope-control-v1",
  "status": "design_only_not_implemented_or_run",
  "date": "2026-10-06",
  "purpose": "scope-word changes under matched opening contexts; no forgetting or general understanding claim",
  "history": "Thank you for your message. Let me address your question.",
  "questions": [
    {
      "id": "delete-partial",
      "verb": "delete",
      "scope": "partial",
      "instruction": "Please help me delete the selected items."
    },
    {
      "id": "delete-whole",
      "verb": "delete",
      "scope": "whole",
      "instruction": "Please help me delete the entire order."
    },
    {
      "id": "remove-partial",
      "verb": "remove",
      "scope": "partial",
      "instruction": "Please help me remove the selected items."
    },
    {
      "id": "remove-whole",
      "verb": "remove",
      "scope": "whole",
      "instruction": "Please help me remove the entire order."
    }
  ],
  "openings": [
    {
      "id": "free",
      "stem": ""
    },
    {
      "id": "delete",
      "stem": " To delete"
    },
    {
      "id": "remove",
      "stem": " To remove"
    }
  ],
  "branches": [
    {
      "id": "explicit",
      "partial": " the selected items",
      "whole": " the entire order",
      "token_lengths": [
        3,
        3
      ]
    },
    {
      "id": "alternative",
      "partial": " some items",
      "whole": " your order",
      "token_lengths": [
        2,
        2
      ]
    }
  ],
  "decision_probes": [
    {
      "id": "after-the",
      "suffix": " the",
      "tokens": [
        "Ġselected",
        "Ġentire"
      ]
    },
    {
      "id": "after-verb",
      "suffix": "",
      "tokens": [
        "Ġsome",
        "Ġyour"
      ]
    }
  ],
  "controls": {
    "generation": [
      {
        "question": "delete-partial",
        "opening": "free"
      },
      {
        "question": "delete-partial",
        "opening": "delete"
      }
    ],
    "score": {
      "question": "delete-partial",
      "opening": "delete",
      "all_branches": true
    },
    "probes": {
      "question": "delete-partial",
      "opening": "delete",
      "both_probes": true
    }
  },
  "expected": {
    "primary_generations": 12,
    "repeat_generations": 2,
    "total_generations": 14,
    "primary_scores": 32,
    "repeat_scores": 4,
    "total_scores": 36,
    "primary_probe_forwards": 16,
    "repeat_probe_forwards": 2,
    "total_probe_forwards": 18
  },
  "generation": {
    "checkpoint": "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537/best.pt",
    "checkpoint_sha256": "2a2e549f954d96325daf4cb5c286edae9f9441b2ea806bab0bcdd1734fa46479",
    "tokenizer": "artifacts/tokenizer/bitext-bpe-v1/tokenizer.json",
    "tokenizer_revision": "43b80bbb10695d80aa442fa8435eb3ca8c33111554bf1d076006dcf2b7d76ea7",
    "model_config": "configs/model.json",
    "device": "cpu",
    "cpu_threads": 2,
    "seed": 42,
    "context": 512,
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
  "evidence_sha256": {
    "docs/condition-use-measurement-v2-results-20261006.md": "0e9b55368aba7d1965f478bbe4a3d49b32b38474fe0f7398cb1ad425b4ffefe7",
    "outputs/condition-use-measurement-v2-20261006/completion-freeze.json": "3641a6d9462db39a060d231f0671fd5c0c62f9ac457a4e5d9574212ff8054c59",
    "outputs/condition-use-measurement-v2-20261006/generations/order-2-h1-q0.json": "4a61a28637d26658d94420534679fbf1f5876848407edfcef4b0e53d59e9329c",
    "outputs/condition-use-measurement-v2-20261006/generations/order-2-h2-q0.json": "75dc8f5e70f992db44ec019d00eb96001dcf98d054f79b8a3828dad4e575508f"
  }
}
```

