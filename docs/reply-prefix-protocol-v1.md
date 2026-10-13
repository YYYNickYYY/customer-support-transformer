# 回复前缀与任务偏好：预登记 v1

2026-10-06。问题：第五模型的细粒度任务偏好是否随回复历史变化，是否出现开头识别诉求而后续转向模板？不训练、不改参数、数据、活动生成配置或话术清单。

## 范围与冻结

沿用condition-use-v1已固定的8条原始问题、4个相反任务配对（部分商品1469/1256、整单824/969、申请16867/16597、资格6461/6688）。原改述和重复控制不加入新的任务样本。保留全部8来源，不按上轮好坏抽选。8参考训练目标不拼进生成输入。

先冻结本协议、机器计划、实现/单测、模型/tokenizer、运行环境和前轮完整结果哈希，后推理。第五best.pt、CPU2/seed42、greedy、角色屏蔽、EOS不屏蔽、context512、**回复总token预算320（含强制前缀）**、no_repeat0。前缀不含EOS或控制token；不静默截输入、前缀或分支。只写独立outputs/reply-prefix-20261006-v1，不读dev/sealed_test，不推送GitHub。

## A：真实轨迹观察与共享前缀条件对照

先完整重放8条空前缀原始生成，逐token/正文/停止与前轮一致后才执行后续。若不一致，整批停止，不临时更换模型或放宽门槛。

每条原始轨迹取0及三个预定阶段：在已生成token数至少32/64/96之后，取**首个**非EOS、可正确解码且正文前缀以句末`. ! ?`结束的token边界。解码前缀必须是完整回复文本的字面前缀；不足或无边界记unavailable，不往前取、不按任务转向位置挑选。不同阶段落同一位置时保留阶段记录但不得当独立观察。精确沿用旧token ID，不把正文重新分词，不含未来token或参考目标。

每个前缀分别在来源问题和配对的另一任务问题下，打分同一对预写分支句。所有分支文本随代码冻结，不从新回复摘最佳区分词：

- 部分商品：`\n\nPlease tell me which items you want to remove.`
- 整单取消：`\n\nPlease confirm that you want to cancel the entire order.`
- 退款申请：`\n\nPlease provide your purchase details to start a refund request.`
- 退款资格：`\n\nThe refund policy explains which situations are eligible for a refund.`

只计完整分支token，不计问题/前缀，不补EOS，不加角色屏蔽到训练式NLL；保存总NLL、token平均NLL及长度。候选长度/措辞不同，不能将分支打分当真实业务概率或解码器只存在这两种选择。句末取样减轻半句话衔接干扰，但不能保证探针句与所有历史自然连贯。

每配对定义A=部分/申请，B=整单/资格：M=`mean_NLL(B)-mean_NLL(A)`，正值偏好A。共享同一历史时的条件分离S=`M(question_A)-M(question_B)`，正值表示按问题作方向性区分。两种来源历史都报告。自轨迹正确任务margin为A问题的M或B问题的-M；报告各阶段相对0的变化，不设事后阈值，不删反向变化。0阶段重复历史是同一测量，不当额外样本。

S保持前缀相同，避免把两条各自不同历史误当问题影响；问题长度/词面/位置仍有差异，不能隔离某词、层或注意力头的因果。沿真实轨迹看到margin变化是关联，历史是问题与模型共同产生的，不能直接声称历史导致任务遗忘。没有强制wrong前缀、空问题、随机词或其它参数实验。

## B：固定前缀干预及续写

8问题各运行empty、neutral、task各一次，另重复首条neutral一次，共25次。empty即A阶段的重放结果，不重复生成。neutral固定为`I'm here to help. `；task为每类固定一句：

- 部分：`I can help you remove specific items from your order. `
- 整单：`I can help you cancel your entire order. `
- 申请：`I can help you submit a refund request. `
- 资格：`I can help you understand refund eligibility. `

这是人工设置回复历史，不是修改问题或模板；不是参考答案注入，但task显式增加正确任务提示，因此改善不能单独证明模型原本理解。neutral/task的BPE长度不同，会改变位置和剩余预算；不得把变化全归因语义，不把结果当已获准的解码改进。

保存前缀token、新生成token、完整回复/续写、角色屏蔽/EOS停止及每步top5。重放empty须完全一致；首neutral与末重复须完整token/正文/停止一致，否则无效。总预算统一320，前缀占用预算，新生成上限320减前缀token数，EOS按全回复计数。生成不使用分支探针、目标、label或intent。

全文AI按condition-use-v1任务标准判match/mismatch/ambiguous；只开场说对但正文整单/进度/资格替代，不通过。强制task文字本身不当正确续写证据，须看新生成的步骤或相关澄清；若仅重复开场/无有效后续，记ambiguous或mismatch并解释。不把关键词检测当语义评分。风险、重复、实体、通顺只做旁注，不回写历史135dev/因素表。待评pending不能当0。

## 决策边界

真实轨迹中任务margin降低而S也降低，提示该探针下条件区分变弱；若S仍正而自任务margin转负，说明有问题影响但共享历史偏好另一分支；两者都不能泛化成“模型完全不看问题”。固定正确task前缀仍转向，说明正确开场不足以稳定正文；若改善，记录可控历史影响，但不采用生产策略，不称机制已唯一定位。两个域/两方向/全部阶段及未改善样本均报告，无总体理解率或显著性声明。

执行后验证权重不变、梯度None、optimizer_updates0与历史依赖/表格不变。新文件创建式写入，同哈希可补缺，冲突/损坏不覆盖；全文评审与数值产物分别冻结。先诊断再讨论后续明确问题，不自动授权第六次训练或补话术。
