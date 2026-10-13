# 第五模型开场范围对照 v1：执行结果

2026-10-06。按[已冻结协议](opening-scope-control-v1-protocol.md)及[有限工程验收](opening-scope-control-v1-acceptance.md)执行。没有第六次训练，没有调整参数、目标文本、tokenizer或原greedy配置。不是业务准确率测评或遗忘机制证明。

## 结论

**问题范围影响了指定短语的相对概率，但没有稳定落实到实际开场对象。** 四个固定开场配对均有正向候选敏感性与局部探针变化；然而绝对偏好受候选措辞影响，而且四组均未达到两侧分别偏好自身候选。不能据此称“已经理解，只是解码错误”。

固定开场的四条部分商品回复：三个首对象为整单，一个purchase范围无法确认，全文四条均冲突。四条整单回复：首对象及全文方向均匹配，但流程仍有账单/发票跳转、缺取消步骤等问题。自由开场仅作参照，不用与强制stem的差异证明因果。

## 执行与控制

- 先prepare冻结协议、准入、源码/单测、真实环境、第五权重、tokenizer、配置、全部token刺激及旧证据/14工作簿；第五checkpoint SHA256为`2a2e549f954d96325daf4cb5c286edae9f9441b2ea806bab0bcdd1734fa46479`。
- 14续写完成，全部EOS，无预算截断；两个重复控制token/文本/停止原因完全一致。
- 阅读完整回复并写入14条AI判读，先freeze-review，之后才执行36候选评分、18独立探针forward。没有根据数值修改语义判断。
- 候选/探针重复控制通过；每阶段model_state_unchanged=true、gradients_none=true、optimizer_updates=0，绑定依赖哈希通过。
- 执行后专项回归35 passed（19.57秒）；已完成generate入口重放只校验、不新增推理。执行器未修改，无需重复整套451项历史回归。
- CPU2、seed42、context512、历史+stem+新回复预算320、greedy、no_repeat0、角色token屏蔽、EOS允许。未读取sealed_test，未推送GitHub，未更改旧表或v2冻结文件。

## 先冻结的语义判读

下表仅12条主体，不把重复控制当独立病例。`matching`只表示范围/任务方向，不保证步骤真实可行。

| 问题 | 开场 | 首个明确对象 | 全文任务 | 主要证据/缺陷 |
| --- | --- | --- | --- | --- |
| delete部分 | free | 无法确认 | 匹配 | purchase范围含糊，随后询问具体待删除商品 |
| delete部分 | delete | 整单 | 冲突 | your order后转delete your account |
| delete部分 | remove | 整单 | 冲突 | your order后混入账户删除/发票查看 |
| delete整单 | free | 整单 | 混合 | 承认整单，却询问articles待删除 |
| delete整单 | delete | 整单 | 匹配 | 删除/确认整单；账单入口未核实，delete or delete重复 |
| delete整单 | remove | 整单 | 匹配 | 整单移除方向，发票查看步骤不能实现取消 |
| remove部分 | free | 部分 | 匹配 | some items及具体商品澄清；末句语病/重复 |
| remove部分 | delete | 无法确认 | 冲突 | purchase范围含糊，后续变成地址修改/订阅 |
| remove部分 | remove | 整单 | 冲突 | 整单＋账户/地址混杂 |
| remove整单 | free | 整单 | 混合 | 承认整单，却询问articles待移除 |
| remove整单 | delete | 整单 | 匹配 | 整单取消方向，混入查看发票 |
| remove整单 | remove | 整单 | 匹配 | 整单方向，给未填订单号占位符且缺取消动作 |

主体首对象：partial 1、whole 9、unavailable 2；全文matching 6、conflicting 4、mixed 2。自由开场两条部分诉求全文匹配，两条整单诉求混合。没有把your或单独order出现当范围判断，purchase不硬判整单；强制stem本身不算生成成功证据。

每条帮助/风险/实体/重复/通顺旁注见[冻结判读](../outputs/opening-scope-control-v1-20261006/semantic-reviews.json)。按风险v1.1、其余v1.0边界文字记录，不新建总分或写回135条因素表。账户删除流程为高影响风险；普通订单/地址导航不自动记风险。含糊语病不能按ensure关键词判保证。问题无具体编号，因此不能声称实体保留能力通过；新增占位符/个人资料要求单独记载。AI知道协议和旧观察，非独立人工盲评。

## 候选与探针：相对敏感性不等于选对

M=NLL(整单)-NLL(部分)，正值偏部分，负值偏整单；D=M(部分问题)-M(整单问题)。下列M使用mean NLL；同一说法内长度相同，total方向一致，不跨3token/2token直接比较绝对总分。

| 问题动词/开场 | explicit M（部分/整单问题） | alternative M（部分/整单问题） | D explicit/alternative | 双方分别选对 |
| --- | --- | --- | --- | --- |
| delete/delete | 1.874515 / 0.012566 | -1.117274 / -5.133698 | 1.861949 / 4.016424 | 否 |
| delete/remove | 1.848839 / 0.057970 | -0.774269 / -4.180471 | 1.790869 / 3.406202 | 否 |
| remove/delete | 1.995877 / 0.203417 | -1.261769 / -4.755929 | 1.792460 / 3.494160 | 否 |
| remove/remove | 2.064199 / 0.304220 | -0.869720 / -3.857782 | 1.759979 / 2.988062 | 否 |

四组D均正向，但explicit两侧都偏部分，alternative两侧都偏整单。正向敏感性不是跨零选择，也不是概率偏好与实际greedy开场一致。delete/remove匹配并未改变这一共同模式；只在这个矩阵成立，不推广到全部措辞。

局部R=log p(部分token)-log p(整单token)：

| 问题动词/开场 | after-the R（部分/整单问题） | ΔR | after-verb R（部分/整单问题） | ΔR |
| --- | --- | --- | --- | --- |
| delete/delete | 3.113908 / 2.494963 | 0.618945 | -2.924079 / -6.788925 | 3.864846 |
| delete/remove | 3.178623 / 2.716830 | 0.461793 | -1.788049 / -5.122704 | 3.334655 |
| remove/delete | 3.811954 / 3.322861 | 0.489094 | -3.087416 / -6.713264 | 3.625849 |
| remove/remove | 4.242182 / 3.724926 | 0.517256 | -1.954070 / -5.059145 | 3.105075 |

全部ΔR正向，after-the两侧都偏selected，after-verb两侧都偏your。指定词在top5外也直接保存概率/log概率，未补零。两个probe是不同上下文，不跨其绝对概率比较。some/your不是完整动作分类器，the是人工干预，不能当实际greedy轨迹。

## 决策与后续边界

本实验收尾，不继续新增病例/候选/规则，不启动训练或强制开场部署。它支持的是“范围改变有可测分布影响，但本矩阵中的偏置与措辞依赖仍占优，实际行动不可靠”，不是“完全没利用问题”或“理解后被解码毁掉”。

若继续，优先**只读对齐现有八条固定stem生成的首次对象分叉**：使用已保存生成token/top5，与本次同位置候选/探针核对；明确实际走到哪些上下文、哪些探针上下文从未走到。目标是区分词汇路径偏置与后续跨任务重组，不再增加生成或调参。若trace缺少需要的概率，只记证据不足，不反推或补0；需要新forward时另预登记后请求执行。

## 产物

唯一新结果目录：[opening-scope-control-v1-20261006](../outputs/opening-scope-control-v1-20261006/)。包含freeze、14生成trace、先冻结的判读、36候选、18探针、阶段status/freeze和[数值汇总](../outputs/opening-scope-control-v1-20261006/summary.json)。完成绑定见completion-freeze.json；不新增Excel汇总。
