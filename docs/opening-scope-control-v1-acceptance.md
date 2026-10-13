# 开场动作范围对照 v1：新干预契约验收

2026-10-06。对应[预登记协议](opening-scope-control-v1-protocol.md)、[执行器](../src/my_project_1/opening_scope_control.py)与[单测](../tests/test_opening_scope_control.py)。

## 验收结论

**通过“等位置开场范围cloze对照”的工程准入。** 仅验证人工固定开场下的编码、分支NLL、局部分叉概率及记录流程；不验证真实客服理解、自然生成历史或遗忘机制。

本轮只使用冻结tokenizer和新建合成oracle/合成EOS模型，校验旧文件哈希。没有把第五权重加载到模型，没有新真实模型forward或生成，没有训练、调参或目标文本修改。实际实验目录未创建，尚未执行prepare或任何真实生成/评分/探针。

原协议内“设计阶段”的状态是当时快照，保留原文与hash；当前实现/验收状态以本报告及项目导航为准。旧v2协议、评分、报告、源码、14个Excel不覆盖。

## 实现内容

- history/stem独立字段及token来源：完整中性历史12token，人工未完成stem为0或2token；生成仅拼接两者，候选和评审标准不进入生成输入。
- 范围对照四问题等长8token；同问题动词的两问题只有第5/6零基token变化，prompt均11token。同一probe配对双方的输入长度/预测位置一致。
- 分支只监督完整范围候选，history/stem/prompt均不计loss、不加EOS、不截断；两组候选分别3/3、2/2。最长分支序列28token。
- probe直接读取两个预指定token的probability/log_probability/raw_logit，保存角色屏蔽后的normalizer、EOS概率和top5。top5未出现仍读取真实值，绝不以缺失补0；log概率用于避免极小概率的数值下溢误解。
- 数值汇总保留每侧M、D、两组说法是否一致、是否双方分别偏好自身方向；等长候选总/均margin的比例关系额外核验。两个局部probe各自保留R与ΔR，不把some/your当范围分类器。
- 分阶段check/prepare/generate/freeze-review/score/probe/summarize。score/probe必须先有完整生成冻结与语义判读冻结。第一对象证据必须包含新生成文字，强制stem本身不能当语义成功证据。
- create-only原子记录、payload/job/freeze绑定、旧记录全预检后才加载模型、相同hash补缺；完整阶段重跑只核验。控制失败/状态改变标invalid并停止，无优化器入口。
- prepare必须读取本报告机器准入receipt并验证协议/实现/单测/共用函数hash；验收后源代码变动不能静默沿用本次准入。真实运行时再冻结全部jobs、实际环境、模型/配置/tokenizer及14旧Excel。

## 测试证据

新方法/执行器 **35 passed，21.58秒**；完整项目回归 **451 passed，384.49秒（6分24秒）**。新专项使用合成模型，整个项目回归包含既有模块测试，不是实际实验运行。

| 检查 | 结果 | 限定证据 |
| --- | --- | --- |
| stimuli/token来源与配平位置 | 通过 | 全部14/36/18预定job展开、来源分开、范围差异位置固定，控制ID无重复 |
| 自然历史与未完成stem | 通过新契约 | 拼接/整段response编码一致，无尾随独立空白；明确不套用旧完整句语义验收 |
| 标签/NLL | 通过合成验收 | 八个问题×固定stem代表组合，冻结BPE下逐token独立scalar log-sum-exp核对；不是第五logits复现 |
| 角色屏蔽与EOS | 通过合成验收 | oracle让控制token原logit占优，独立手算屏蔽normalizer；EOS不屏蔽 |
| top5外的指定token | 通过 | oracle强制指定token低于top5，仍保存非零概率与log概率；不补0 |
| 相同分叉预测位置 | 通过 | after-the/after-verb与对应分支token的log-odds交叉核对；允许token共用normalizer取消，但不把原始NLL当被屏蔽分布 |
| 未来泄露/异常/长度 | 通过 | 因果oracle未来变化不影响先前分叉；非有限输出/控制目标/溢出拒绝；成功与异常后eval模式恢复、梯度None |
| 回复语义证据边界 | 通过结构验收 | 证据可跨stem与新生成，但不能仅引用强制stem；没有自动业务语义评分器 |
| 冻结、补缺、完成重跑 | 通过合成流程 | 只在测试临时目录模拟中断，旧记录bytes/mtime不变；14续写/36评分/18probe完整流程通过；损坏拒绝且不加载权重 |
| 准入门控与绑定 | 通过 | 无有效验收不创建prepare目录；准入绑定变动即拒绝；numerical stages无评审冻结时不得加载模型 |
| 真实模型运行 | 未执行 | 无第五模型真实回复、分支或probe结果，无能力分数 |

测试使用autouse拦截torch.load；真实checkpoint仅做流式SHA256校验，未反序列化到模型。合成模型使用新建参数，均未做梯度或optimizer更新。临时目录的合成freeze/records不是项目真实实验产物。

## 解释边界与下一步

问题是合成的delete/remove表述，explicit候选重复了问题措辞，alternative仍有词频/句法影响；两个probe尤其some/your并不是完整业务动作。独立语义金标准未建立。人工stem改变输出分布，不是建议强制开场上线，也不能用free与cloze的差异证明长度导致遗忘。

下一步若用户确认运行，只能先prepare冻结当前协议/准入/实现/环境及全部token刺激，然后按协议完成14次续写，阅读全文冻结语义判断，之后36条候选评分及18次probe。控制/依赖失败即停，不换问题、候选或开场。保持第五权重和原greedy配置，不训练或调参。

**本轮止于实现与方法验收；实际实验尚未启动。**

## 机器准入receipt

```json
{
  "status": "accepted_for_matched_opening_scope_cloze_only",
  "date": "2026-10-06",
  "fifth_inference": false,
  "training": false,
  "tests": {
    "new_method_and_runner": 35,
    "method_result": "35 passed in 21.58s",
    "full_project": "451 passed in 384.49s"
  },
  "sha256": {
    "docs/opening-scope-control-v1-protocol.md": "5534fb0bce3accbe2fe3be6c275a7d300037f2160a0276f8b30b5fedc818ce9f",
    "src/my_project_1/opening_scope_control.py": "4617f01153b4f42e3da4b35c502a647b423e7e6cc3d4826005d19255dfea944c",
    "tests/test_opening_scope_control.py": "9f2c5fe2001ba8133e268095e926cc418e332e963459587be8e4d21497a94b88",
    "src/my_project_1/condition_measurement_v2.py": "a1ae29881cc8a752633e7178bf3fd4a9a211ce6fa7a0d5b1aad7ad5bb135ab0f",
    "src/my_project_1/reply_prefix.py": "db77bcf126ab0e1600fac57df4db33f018118ea27712d9eb526ab9eff1805213"
  },
  "limitations": [
    "synthetic matched questions",
    "forced incomplete stem",
    "wording/copying effects",
    "no independent business construct",
    "no forgetting claim"
  ],
  "real_execution_registered": false
}
```

