# 训练输入—目标一致性审计：预登记 v1

日期：2026-10-06。问题：第五次模型训练文本是否提供了编号保留、否定条件、部分商品操作、退款申请步骤的正确监督？这是数据支持证据审计，不是第六次训练，不以逐条修复已知生成病例为目标。

## 来源和顺序

唯一主数据为第五次实际使用的 `data/derived/refund-execution-v3.1/train.jsonl`（SHA256 `4d1ed247e99b7b9aa5914be35ea9e6bcb8d4a8f70f1188808cda5b8bd07fbe73`）。用对应 run 的 experiment/config 和 revision manifest 验证链接。后续补充清单、原始 train、dev、sealed_test 均不读取。先测试纯函数，再冻结协议、计划、实现、测试、来源和既有诊断文件哈希，之后才扫描训练正文。

固定第五次 run `20261005T132346-full-ebf624ab40734b858b46553b4da37537`，无需加载权重或运行生成；既有输入敏感性诊断只是提出问题的来源，不混入训练样本。

## 候选规则及固定抽样

具体正则和域定义随 `src/my_project_1/input_target_audit.py` 冻结；输入筛选不查看目标好坏。

1. 编号：order/invoice 后紧邻可选 number/no/id/# 和至少三位十进制数字；保留前导零。日期、金额、步骤序号不算该规则编号。所有命中行记录准确字面保留和目标占位符，不自动判错：自然客服未必需要复述编号。显式要求 include/repeat/mention/copy 编号另列候选，全文确认要求的对象。
2. 否定：直接拒绝 cancel/request/get refund 的窄规则单列；do not/don't/not/cannot/can't/no longer 为宽筛查。不能把“不知道如何申请”“无法付款”当作拒绝申请；“不再想要邮件”可能是退订正例。零窄命中只说明该规则未找到，不能证明所有语义否定不存在。
3. 部分商品：输入 remove/delete + item/product + one/some/several/single/part 等部分信号。不凭 change_order 标签或目标包含 order 就判整单。全文区分部分删除、澄清具体商品、整单取消、无关路径及含混。
4. 退款步骤：输入 refund + how/steps/request/apply；域为 get_refund/check_refund_policy/track_refund。全文区分可用申请路径或相关必要澄清、只讲资格、进度、混合及含混。资格内容伴随有效申请路径可接受，不以出现 eligible 自动判错。

六个候选层（identifiers、explicit_copy、direct_negation、broad_negation、partial_items、refund_steps）各最多8条；四个域对照层（identifiers_control、negation_control、partial_control、refund_control）各最多4条。对照为同一预定域中对应主信号未命中者，不声称全数据随机对照。域：编号 order/invoice 输入主题；否定 cancel_order/get_refund；部分 change_order/cancel_order；退款上述三 intent。宽否定从全 train 筛查。

每层按 SHA256(`train-input-target-audit-v1:42:<stratum>:<完整ID>`) 升序，ID作平局顺序；不足取全部、报告不足，不按结论补样。跨层可重复入选，阅读全文按ID去重，各层仍保留分母。样本规模先固定，不能见结果后换规则、扩大层或挑支持猜想的例子。

## 判读与报告

全量只输出结构/词面统计及 intent 分布；人工AI复核固定样本完整输入和目标，记录原文证据、aligned/mismatch/ambiguous/not_applicable 和理由。未评为 pending，不当0；关键词不能自动赋语义结论。编号没复述不能一律算错，显式复制请求和普通保留分别判断。报告支持性样本和反例。

各层分母为其选定且完成复核的行；样本结论不得冒充全 train 缺陷率。无总体质量总分，不与生成评分表合并。窄规则漏检、英文模板语料、样本量、AI非独立盲评是限制。当前审计只检查存储文本；不预登记 token 截断归因，后续若需检查另立阶段，不能以当前全文正确证明模型看见全部目标。

## 决策边界

- 输入条件缺少训练支持：记录覆盖不足证据，先设计有代表性的独立诊断/数据方案，不直接加参数。
- 输入与目标错配：数据可能贡献问题，但不能证明某条训练样本导致生成错误；不自动产生改写清单。
- 样本目标正确而模型失败：转向条件学习/生成机制诊断；不能因有限样本通过宣称数据无问题。
- 筛查或复核未完成：明确保留 pending，不提前决定因素实验。

审计产物写入独立目录；源数据、模型、训练/生成配置、历史评分、表格不变。不训练、不推送GitHub。
