# 退款保证覆盖补充清单协议 — 2026-10-06

本清单仅针对第五次冻结训练目标 `refund-execution-v3.1` 的遗漏，不启动训练、不切换活动配置。重复评分先完成，再冻结候选、审阅及批准修改。旧评分、旧训练目标和程序保持原样。

## 提名范围

扫描全部 train，纳入退款 intent，或回复含 refund/reimbursement/rebate/compensation/restitution 的记录。逐句召回：

- ensure/guarantee/promise 与 resolution、resolved、到账、完成、处理、in order 的组合；包含不在同句出现退款名词但上下文为退款的保证。
- ensure ... receive ... refund/reimbursement/rebate/compensation/restitution。
- will be processed/issued/completed/credited、you will receive/get 等确定退款结果。
- actively/already ... processing/issuing，补回 compensation 的已执行声明。

词法仅提名，不自动批准。assist/help、work together、may 等出现在其他分句，不能自动豁免确定保证。每条审阅记录原句、上下文、理由及保留/批准结论。若召回混入预计时间、查询准确性、普通页面、非退款动作等，不顺便修改。有限召回不能证明 train 中所有语义变体已经消失。

## 改写边界

只整句改写退款执行、到账、补救结果承诺为核查、请求或协助表述。保留原句正常功能、已有条件、编号/标题和实体；不能新增资格/期限，不把整个句子粗替成重复的通用指导。普通服务意愿、明确努力限定、宽泛满意/体验目标保留。仅修保证，不把语病、去冗余或实体恢复混进因素。

核查完整前后文：确认相邻澄清及支持句没有新增接缝、重复核查指示；既有 v3.1 的批准内容不倒退。批准片段全部处理，并检查同一批准原句在提名范围内的所有出现，不按单个 ID 挑选。

## 冻结与验证

候选及材料先存 SHA-256，显式决策后存批准清单、逐记录前后全文及最终哈希。重放只能改变批准的 response 片段；ID、顺序、非 response 字段、实体多重集及编号标题保持不变。冻结 tokenizer 下检查非零回复监督和截断。绑定 dev、tokenizer、模型、生成、第五次训练配置的哈希，结束时复核未改。

本阶段只产清单及重放验证，不修改活动 train、学习率、轮数或模型，不生成训练启动任务，不读取 sealed_test。AI 审阅不等同独立人工认证。下一次是否训练需另行决定。
