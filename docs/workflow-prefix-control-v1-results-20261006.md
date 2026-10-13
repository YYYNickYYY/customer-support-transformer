# 第五模型流程前缀对照 v1：执行结果

2026-10-06。依[预登记协议](workflow-prefix-control-v1-protocol.md)与[有限准入](workflow-prefix-control-v1-acceptance.md)执行。协议与准入中的“方法阶段/未运行”是冻结历史状态，不覆盖原文；当前结果以本报告为准。

## 回答实验问题

**问题条件能影响部分后续对象，但不能稳定约束下一动作与全文任务。账单领域线索在这对固定问题中表现较差，但任务重组也发生在中性/订单前缀下，不能把失败专归账单。**

- 中性两措辞下，部分商品回复均维持定位商品/删除商品；整单回复却都在定位订单后要求Download，再以泛取消尾句结束。
- 账单A两条均mixed：部分商品先定位正确商品再混入Add New Address；整单定位订单后混入Download，再泛称取消。
- 账单B两条均conflicting：部分诉求转Download/Save（对象含糊），整单诉求明确转发票Download。四条账单续写没有explicit_return或implicit_return，不表示总体纠偏率为零。
- 订单前缀也不是稳定解法：A部分商品混入地址，B部分商品转Delete Account；B整单仅定位/查看订单和泛取消建议，下一动作不足以判断。

在部分商品问题上，账单A/B全文都比各自中性条件差；但A与订单A同类混杂、B与订单B都冲突，不能说账单是唯一干扰。整单A的账单和中性续写完全相同，B账单才明确变成发票下载。结果有措辞依赖，没有两问题跨两措辞一致的账单特异性结论，不估计统计显著性。

## 冻结、执行与控制

prepare冻结协议、准入、源码/单测、环境、权重/配置、tokenizer、全部14个token刺激和历史14Excel。第五checkpoint SHA256为`2a2e549f954d96325daf4cb5c286edae9f9441b2ea806bab0bcdd1734fa46479`。

CPU2/seed42、greedy、context512、no_repeat0、角色token屏蔽/EOS允许；每条输入49token（prompt11+prefix38），历史+回复预算320，续写预算282。没有候选评分或probe，没有改问题/前缀/判据。

14生成完成（12主体+2重放），全部EOS，无截断。两个控制token/正文/停止一致，top5顺序一致、概率容差通过；model_state_unchanged=true、gradients_none=true、optimizer_updates=0，依赖哈希通过。

阅读全文写14条AI辅助语义判读，经freeze-review验证后才summarize。评审知道协议与旧结果，不是独立人工盲评。没有自动按关键词判业务正确；没有参考loss替代内容判断。

## 判读边界：避免把order当成功

本协议忽略登录、订单定位/查看等纯准备动作，判断首个明确实质推进动作。先定位订单并不意味着正在删除整单，也不能因此把部分诉求判错；partial-neutral-b随后明确删除desired items，判匹配。

Download是具体错误动作，但it/details没有说明所下载的对象时，next_object记unavailable，next_action仍可conflicting。不可从强制前缀invoice补出模型实际生成的对象。whole-order-a明确写Download next to the order，保存whole对象/错误动作；whole-billing-b后续明确invoice才判other。

whole-order-b全文保持取消方向，但新续写只有订单查找/查看和Follow prompts to complete cancellation，不把泛尾句当有效下一取消动作。其next_action/next_object均unavailable，full_task为matching；这不等于任务成功。全文mixed保留错误步骤与相关任务尾句的并存，不能因为末句提取消就洗掉Download。

## 12主体逐条件结果

| 领域/措辞 | 部分：下一对象/动作；全文 | 整单：下一对象/动作；全文 | 配对状态 |
| --- | --- | --- | --- |
| neutral/A | partial/matching；matching | unavailable/conflicting；mixed | not_both_match |
| neutral/B | partial/matching；matching | unavailable/conflicting；mixed | not_both_match |
| order/A | partial/matching；mixed | whole/conflicting；mixed | not_both_match |
| order/B | other/conflicting；conflicting | unavailable/unavailable；matching | unavailable |
| billing/A | partial/matching；mixed | unavailable/conflicting；mixed | not_both_match |
| billing/B | unavailable/conflicting；conflicting | other/conflicting；conflicting | not_both_match |

没有both_match配对，五组not_both_match、一组unavailable；无法判断不算失败或成功。主体next_action为4matching/7conflicting/1unavailable；全文3matching/6mixed/3conflicting。这些只是固定病例描述，不是准确率，也不把2条重放当独立样本。

summary中的pair_status_agreement只说明粗配对状态一致：neutral=true、billing=true、order=false。**billing的true不代表两措辞生成机制或缺陷相同**：A为mixed，B为conflicting。保留每侧标签与完整证据才能看出差异。

## 帮助及副作用

每条旁注见[冻结判读](../outputs/workflow-prefix-control-v1-20261006/semantic-reviews.json)，只按既有风险v1.1/其余v1.0文字记录，不创建五项总分或新Excel。

- 相关商品定位仍可提供帮助；Add New Address、Delete Account、Download不能替代移除/取消动作，后续路径需业务核实。
- partial-order-b明确自助Delete Account是高影响账户控制风险；should find不能免除该风险。普通订单/地址/发票导航不自动触发风险，未触发不等于已核实安全。
- 题目没有真实订单号；新增{{Order Number}}是未填占位符，不是实体复制通过，也不是丢失既有编号。这个实验不检验具体实体保留能力。
- partial-order-b有Delete or Delete、Delete Account or Delete Account重复；whole-order-b重复view your order。账单B两条下载类步骤冗余且should be able to Download/Save/View button有明显语病。账单A部分accuracy句冗余且不顺。
- 全部EOS，不把这些缺陷归于长度截断；EOS也不证明回复正确。

## 验证及产物

执行后新专项30 passed（5.18秒）。完成generate入口重放仅验证哈希、没有新推理。旧开场冻结/权重/旧表/当前实际train哈希核验通过；无第六次训练、参数/数据变动、sealed_test读取或GitHub推送。

唯一新产物目录：[workflow-prefix-control-v1-20261006](../outputs/workflow-prefix-control-v1-20261006/)；含freeze.json、14生成trace、generations-status/freeze、semantic-reviews/review-freeze、[summary.json](../outputs/workflow-prefix-control-v1-20261006/summary.json)、completion-freeze.json。完成绑定覆盖本报告及阶段证据，不修改协议/准入/执行器。

## 决策

本实验收尾，不继续换前缀/病例直到出现好结果。当前证据不支持“账单线索就是唯一原因”，也不支持“正确订单前缀即可修复”，不自动改学习率/轮数/模型或补话术。

若继续，下一步应先对本次已有trace做只读共同历史分叉核对：找出问题/领域变化何时从相同续写切换到Delete、Download或Add New Address，不用不同上下文的概率作同位置对照。这个核对可进一步界定下一实验是否有明确可测问题；若需要新forward，另预登记和授权。不以单条坏回复修复作为学习实验的目标。
