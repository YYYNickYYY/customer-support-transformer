# 训练数据质量审查协议 v1

日期：2026-10-04。目的：复核第二次评估新增的7条无依据风险，并检查对应训练话术。
这是诊断审查，不是重评全部27条，也不修改 v1.0 标准、历史分数、训练数据或模型。

## 预先固定的抽样规则

1. 目标intent由已有配对评分中 `unsupported_claim` 从0变1的样本确定，不从train话术挑选。
2. 七类为 cancel_order、change_order、edit_account、get_refund、place_order、registration_problems、track_order。
3. 仅读取冻结 `data/splits/bitext-v1/train.jsonl`，验证其SHA-256与tokenizer配置一致；不读取sealed_test。
4. 每类将样本按 SHA-256(`train-quality-audit-v1:42:` + 样本ID) 升序排列，同哈希按ID排序，取前5条；不足5条全取。
5. 在显示任何抽样话术前，保存35条ID清单、来源哈希、规则、seed=42和清单哈希。后续不得替换“不支持猜想”的样本。
6. 完整阅读抽样问题和回复。每条人工审查记录具体原文、服务意愿/体验话术/结果保证/业务断言类型及缺陷。
7. 可以对七类全部train做预定义词法筛查，但只报告“候选命中率”，不称实际风险率，不以规则命中自动赋予v1.0分数。

## 全量词法筛查的预定义规则

- guarantee_or_promise：`guarantee`、`guaranteed`、`guaranteeing`、`promise`及其复数/过去式词形。
- ensure_smooth_or_satisfaction：同一句中 ensure/ensuring/ensures 与 smooth/seamless/satisfaction/satisfactory 同时出现。
- refund_action：同一句中 issue/process/provide/receive/get（含常见词形）与 refund 同时出现。
- uncertainty_or_goal：同一句中 goal/aim/strive/try/do our best/committed/work towards/may/might/if/eligible 出现。
- extra_phrase_counts：固定短语 `ensure a seamless experience`、`ensure a smooth`、`ensure your satisfaction`、`issue a refund`。只作可追溯语言分布统计。

规则为宽松候选筛查。条件性说明、否定、提问、服务意愿均可能命中；不能等同于保证成功或已经办妥。
对35条样本的手工类别采用：服务意愿、宽泛体验/满意话术、具体结果保证、具体业务事实/流程。类别可同时存在，边界必须保留理由。

## 复核与归因

- 对7条读新旧完整生成与参考文本。按v1.0区分服务意愿与结果保证；同类新旧措辞同尺度审视，不只看新增标记。
- 结论采用“明确符合v1.0风险”“明确评分漂移”“边界争议，无法排除漂移”。不修改历史工作簿。
- 如果需要正式纠正，须另存完整配对复核评分并统一审视两次27条，不把局部建议混入旧汇总。
- 训练话术出现相似表达只支持“存在训练来源”的解释，不能证明某一句被记忆、训练污染或因果关系。
- 分开记录：训练文本/模板本身问题；模型独有的拼接、重复或任务偏移；输入实体丢失；生成320token上限与训练512上下文截断。
- 不凭这次审查更改第三次训练因素。先明确评分边界，再决定是否需要训练回复干预。
