# 第四次训练目标干预协议 v2

2026-10-05，版本result-privacy-target-v2。唯一因素：同类解决/订阅取消结果及隐私处理保证的覆盖与完整句改写。第三次冻结v1为基线，不覆盖v1；原train/dev/tokenizer不变。不扩展到订阅成功/投递保证、支付/配送/账户权益/自助操作等其他业务断言。不读取sealed_test。

## 规则与人工确认

先输出有限明确句型候选，在冻结前核查唯一before→after整句及上下文。每个修改保留ID、原/新完整回复、位置和规则。冻结规则记录为完整原句→完整新句，训练时严格重放，不把宽松关键词当风险分。

- 隐私：明确的信息/data/details/email宾语或其代词，未来will/we'll或promise/guarantee引出的handle/treat/keep/store/protect/safeguard/secure等处理保证，并含care/confidential/security/safe/protection等语义。处理投诉/request的care不纳入；secure password建议不纳入。没有确定处理结果的prioritize/committed/重要性态度不改。段落混合非隐私动作的句子拒绝自动整句处理，留作排除记录。
- 解决结果：ensure/guarantee/promise明确指向resolution、resolved或successful outcome。goal/aim/strive/try/may/might、do/doing ... best/everything ... can、work/working ... towards/closely/with/together、help/want to ensure等限定服务意愿保守排除，不改宽泛体验/满意/process目标。
- 取消结果：明确ensure/guarantee/promise使subscription canceled/cancelled、unsubscribed、removed/taken off mailing list、取消成功/及时处理请求；只限订阅取消语义，排除do best等努力限定。不把所有含unsubscribe的正常导航都替换。
- 诊断已确认的5处v1接缝用特定完整句修正，明确we动作，避免work to work、resolution process目标搭配及轻微冗长。这些修正仍属于同一保证话术改写方法，不清洗其他语病。
- 安全开户的明确set up smoothly and securely保证纳入隐私保证上下文完整性处理；不改泛用seamless experience，也不改完整账户业务流程。

整句涉及必要的邮箱/姓名/订单细节询问时保留原询问信息，只替换确定保证分句；其余原有动作、实体、字段均保留。无法可靠分开目标保证和其他内容的句子不纳入，公开排除，不追求消除全train所有风险。

## 冻结/训练

v2目录保存train.jsonl、changes.jsonl、rules.json、controls.json、manifest.json。manifest绑定原train、v1基线manifest/hash、协议/代码/模型/tokenizer/dev/生成配置和第四次训练配置。全部非response字段、行数、ID/顺序不变；校验每个approved原句命中全部处理、无重叠、精确重放一致、修改长度及监督范围。存不存在就拒绝覆盖，manifest最后写入。

未批准候选不属于冻结规则。controls按SHA256(target-controls-v2:42:ID)固定选择，保留服务意愿与密码建议等反例。逐句AI核查不是独立人工风险评分；不将词法候选比例称风险率。

仍5轮、lr0.0003、batch8、seed42、weight_decay0.01、clip1、cpu_threads2。模型/tokenizer/上下文及生成greedy/320不变，重新初始化，不从第三次checkpoint续训。第四次只通过VS Code任务由用户启动，不启动Codex后台进程。后续同27ID风险v1.1/其余v1.0评分，不预填第四次结果。
