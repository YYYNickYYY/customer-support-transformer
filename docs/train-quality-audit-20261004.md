# 训练数据质量审查 — 2026-10-04

## 结论

**不能把历史“无依据风险13/27→18/27”直接解释为第二次训练更爱编造业务事实。**
新增7个标记中，2条在v1.0严格锚点下有明确支持，1条有明确的新旧判定不一致，4条存在服务意愿与结果保证的边界争议，无法排除尺度漂移。
之前据此确定描述“风险增加”的解释过强，应改为“历史风险标记增加，但比较有效性存疑”。历史分数没有被覆盖，也未虚构校正后的总数。

训练数据确实包含大量顺利/满意体验话术，并存在实体替换、重复索取和语病。这支持数据源风格和缺陷可能影响生成的解释，**不证明某条生成被特定样本直接记忆，或训练轮数导致真实业务风险上升**。

本次只审查，不改数据、模型、解码设置、v1.0或两次旧评分，不启动训练，不读取sealed_test。

## 1. 新增7条风险的原文复核

每条均读取客户问题、新旧完整生成及参考文本；参考文本不是业务事实依据。
以下结论只审查新增标记的合理性，不是对两次27条全部风险项的正式重评。

| intent | 关键新旧证据 | 复核结论 |
| --- | --- | --- |
| cancel_order | 旧：`We are here to assist you every step of the way.` 新：`ensure a smooth cancellation experience` | 新句明确确保取消体验顺利，按v1.0确保顺利处理的严格口径可保留1。没有声称已取消，也没有已核实真实取消权限。属于流程/体验保证，不是退款到账承诺。 |
| change_order | 旧：`This will help us ensure that your purchase is processed promptly and efficiently.` 新：`we are committed to ensuring your satisfaction` | 边界争议。新句在committed目标语境，旧句也涉及确保订单处理。只标新句不足以形成一致的差异；没有新增政策、权限或具体时限。 |
| edit_account | 旧：`ensure a seamless experience` 新：`ensure a seamless experience for you` | **明确评分漂移**：同一核心措辞旧0新1。旧回复还有ensure smooth等表达。新语病/泛化指导属于其他维度，不足以把同类话术判为新出现。严格口径应两侧同审视，不能单改一侧。 |
| get_refund | 旧：`we are committed to resolving this matter for you` 新：`ensure a smooth and satisfactory resolution`、`ensure a smooth resolution` | 符合v1.0明示的确保解决结果锚点，可保留新增1。与commitment协助愿望相比，新句保证解决效果。但仍不等于保证一定退款或到账日期。 |
| place_order | 旧：`Our goal is to ensure a seamless experience for you.` 新：`This will allow me to guide you through the process and ensure a seamless experience.` | 边界争议。旧句明确goal，新句也在收集商品信息、便于指导的目的语境。没有订单成功、付款或时间保证。新1仅依据ensure证据偏弱，服务意愿解释更合理。 |
| registration_problems | 旧：`investigate and resolve the issue promptly` 新：`committed to resolving this matter promptly and ensuring a smooth registration experience` | 边界争议。两侧均为调查/commitment和promptly，无注册必成功或截止时间。v1.0明确不因调查中的promptly自动记风险，新句的体验愿望不能直接当成功保证。 |
| track_order | 旧：`we work together to resolve this matter` 新：`I'm here to guide you through the process and ensure you have a seamless experience` | 边界争议。新增是宽泛体验目标，没有新增物流状态、必到或精确到货时间。新旧都重复索取已有订单号，交互缺陷成立，但不是新增业务事实风险的证据。 |

**重要限制：不能简单减去1条或4条，就得到“修正后的风险总数”。**
例如edit_account可以严格解释为两侧均1，也可以目标解释为两侧均0，改变哪侧都会影响汇总；还存在其他历史风险项和从1变0的2条未全面重审。
如需发布校正比较，须统一复核两次全部27条并另存，保留原分与差异理由。
v1.0同时列了严格保证锚点和服务愿望例外，对泛泛seamless体验的优先级仍有歧义，不能声称这次已经排除了所有漂移。

## 2. 固定抽样和来源

预先协议：[训练审查协议v1](train-quality-audit-protocol-v1.md)。在查看train话术前已保存35个ID及清单哈希。

- 仅读冻结train，源revision为`fb86d1b60038970d79fefbfa0c28b47c22d8b961`。
- train SHA-256：`2a5186ebf69c691cb7848c6bd5509882158ff49477e1e99fdb3d0991df968396`，与冻结tokenizer配置一致。
- 七个目标intent来自已有0→1的配对记录，不以train文本筛选。
- 每类按SHA-256(`train-quality-audit-v1:42:`+ID)升序取前5条；不看回复、长度、保证词或分数选样。
- 35条全部完整阅读并记录原文证据；未替换无保证话术的样本。记录包括合理服务回复和可疑话术，不仅包括支持猜想的例子。
- 35条只能作固定诊断样本。没有由此推断所有train实际风险率，更没有将它当线上效果估计。

## 3. 对应train话术分布

另外对七类全部5,614条做预定义词法筛查。下表是**同一句中ensure/ensuring/ensures与smooth/seamless/satisfaction/satisfactory同时出现的候选行数**，不是人工风险分数。
明确goal、if或committed等条件/目标表达也可能命中。refund_action也包括“请求退款”及条件退款，不表示一定退款。

| intent | train条数 | ensure体验/满意候选行数 | 候选命中率 |
| --- | ---: | ---: | ---: |
| cancel_order | 815 | 432 | 53.0% |
| change_order | 791 | 285 | 36.0% |
| edit_account | 785 | 225 | 28.7% |
| get_refund | 812 | 273 | 33.6% |
| place_order | 787 | 335 | 42.6% |
| registration_problems | 814 | 360 | 44.2% |
| track_order | 810 | 229 | 28.3% |
| 合计 | 5,614 | 2,139 | 38.1% |

完整固定短语和候选筛查统计保存在`outputs/train-quality-audit-20261004-v1/lexical-statistics.json`。
七类中`issue a refund`精确短语仅命中change_order的1行，其余为0；这不等于没有退款保证，可能使用rebates/compensation、其他动词或条件表达。
因此，能确认“顺利/满意话术常见”，不能据本规则确认“一定退款也同样常见”。

35条手工审查中：19条包含宽泛体验/满意话术，5条包含较具体结果保证，15条包含服务意愿，类别可重叠，**不作风险率**。
其中同句ensure词法候选15/35，全部候选规则并集16/35。人工类别还包含词法漏掉的`make sure`、rewarding、补偿/返款等形式，不能用自动命中代替阅读。

### 抽样正反证据

下表ID尾号对应冻结revision的`bitext:fb86d1b60038970d79fefbfa0c28b47c22d8b961:<尾号>`，完整ID和全部35条证据另存。

| ID尾号 / intent | 原文证据 | 含义 |
| --- | --- | --- |
| 806 / cancel_order | `ensure a smooth cancellation experience` | 新生成取消回复的同类话术确实在train存在，不证明直接复制此行。 |
| 492 / cancel_order | `ensure a seamless cancellation experience` | 同类取消体验话术；回复中两个操作候选都是同一占位符。 |
| 1141 / change_order | `Our main goal is to ensure your complete satisfaction` | 明确goal，是服务目标锚点，不能见ensure就判保证。 |
| 1234 / change_order | `ensure that your order is updated according to your preferences` | 较具体的修改结果保证，未核实订单可改状态。 |
| 14490 / edit_account | `ensuring that your account details are accurately updated` | 具体资料更新效果保证，同时有不要分享个人标识的安全提醒，证据不能只截有害部分。 |
| 16468 / get_refund | `ensuring that you receive the appropriate compensation for your purchase` | 较具体补偿结果保证，未给真实退款资格依据。 |
| 16771 / get_refund | `assist you in submitting your request` | 只协助提交，不保证审批或到账，是反例。 |
| 18989 / place_order | `committed to ensuring a seamless and rewarding experience` | 与新模型体验话术同类，但仍存在目标/保证边界。 |
| 19763 / place_order | `ensure that I give you accurate guidance` | 收集商品信息用于指导，不能自动当订单必成功。 |
| 21306 / registration_problems | `committed to ensuring a smooth and seamless sign-up experience` | 相似commitment与smooth组合在train存在；开头`contacting to us`也有语病。 |
| 21780 / registration_problems | `so that we can address it promptly` | 无确定时限的协助愿望，promptly不是自动风险。 |
| 25243 / track_order | 问题`order 00004587345current status`，回复用`{{Order Number}}`并要求`provide me with your purchase number` | 训练文本已存在实体替换和重复索取，不能完全归因模型新造缺陷。 |
| 24954 / track_order | 已给订单号后问`confirm the platform or website` | 有新增信息的合理追问，是重复索取的反例。 |
| 24933 / track_order | `ensure ... a seamless experience and all your concerns are addressed` | 除体验外又保证所有疑虑都处理，强于仅“我来协助”。 |
| 19567 / place_order | `loyalty programs, referral programs, and special promotions` | 模糊earn产品问题被引入商户项目，欠真实资料支撑；不证明项目实际不存在。 |

## 4. 缺陷归因与相应措施

归因不是互斥分类，也不等于已识别因果。以下把直接证据与推断分开。

| 问题 | 已见证据 | 能确认/不能确认 | 对应后续措施（本次未执行） |
| --- | --- | --- | --- |
| 顺利/满意话术 | 全量候选2,139/5,614，抽样19/35含宽泛话术，新旧生成同类表达 | 确认训练语言来源存在；不能确认多训两轮造成真实业务风险增加，评分边界仍不稳定 | 先统一“服务意愿/具体结果保证”边界并完整配对复核；再决定是否清理话术，不按ensure全删 |
| 具体业务事实/处理保证 | 抽样修改订单、更新账户、补偿及商户项目等断言，无正式商户证据 | 确认训练目标含未绑定业务依据的断言，不证明事实为假或有真实可用权限 | 未来另建有版本的目标文本，区分提交/核查/批准，接入正式业务资料前不保证结果 |
| 语言/模板缺陷 | train的account!I、contacting to us、help to us、生硬aware of，以及取消入口占位符重复 | 这些形式在训练文本直接存在，可以提供风格来源解释，不能认定全部生成语病均由这些行造成 | 未来清理目标语病和重复模板；保持原始冻结文件不变，衍生版本可追溯 |
| 模型拼接/任务偏移 | 新PIN回复`Once you've entered your identity is verified`、`don't see it in your PIN code`；账户取消被答为cancel subscription | 确认输出有拼接异常和任务混淆，且两条都是EOS不是长度截断；未抽PIN/注销的train，不能断言该问题只来自模型 | 针对这两类另固定抽样查来源，设计最小对立问法测试；不要先扩大模型或强行归因为数据 |
| 实体保留/重复索取 | train 25243替换00004587345再要编号；两次dev发票问题37777、参考回复及模型却用Invoice Number；新旧track_order再要已有号码 | 确认源问答已有实体对齐缺陷，且模型也未保留输入；不证明没有其他机制原因 | 未来采用一致匿名化/实体对齐衍生训练版本，并增加号码保留测试；不能只删礼貌词解决 |
| 生成长度上限 | 两次check_refund_policy均max_new_tokens=320；第二次提示11token+生成320=331，低于512；参考完整串389 | 确认本次末句不完整由新token限额触发，不是达到上下文512；长模板可能促进冗长但未证明是唯一原因 | 如检验生成预算，另做仅改变max_new_tokens的推理实验，不与第三次训练因素混改；或另做目标精简实验 |
| 训练序列截断 | 固定35条完整序列52–270token，0条超过512 | 这35条未因512上限截断；不能据此声称全train无截断 | 只在需要时引用冻结长度统计或单独检查全train，不以生成截断替代训练截断解释 |

## 5. 下一步优先级

1. **先校准评估，不急于第三次训练。** 对两次27条统一复核无依据字段，明确泛泛体验目标与具体结果保证的界线。若修改v1.0定义，另存新版本，双方同标准重评；若只是纠正执行错误，记录v1.0复核差异。
2. 评分可比后，才决定第三次是否只干预训练目标话术。不能把宽泛服务愿望与商户支付/配送/退款事实都按一个正则删除。
3. 实体保留和生成预算是另外两个因素，应分开实验，不与数据清理同时修改，避免无法解释结果。

## 6. 文件与验证

- 协议：`docs/train-quality-audit-protocol-v1.md`。
- 可复现脚本：`scripts/audit_train_quality.py`。冻结、统计、显式审查汇总分阶段，已有输出不会默默覆盖。
- 本次产物目录：`outputs/train-quality-audit-20261004-v1/`。
- `selection.json`/`freeze.json`：35条冻结ID、源哈希、协议哈希；`samples.jsonl`：完整抽样问题/回复。
- `lexical-statistics.json`：七类全train的词法候选计数；`samples-reviewed.json`：35条显式手工类别与原文证据；`risk-review.json`：新增7条的新旧证据、边界和漂移结论。
- 所有35条手工摘录与14条新旧摘录已逐项验证为源原文子串。5项测试覆盖候选而非评分、同句条件、条件退款、服务意愿及内容无关ID排序。
- 未调整历史风险计数，未生成未经全面复核的校正汇总，未读取sealed_test，未修改冻结数据或训练配置。
