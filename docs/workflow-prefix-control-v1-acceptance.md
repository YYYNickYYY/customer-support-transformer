# 流程前缀对照 v1：方法验收

2026-10-06。对应[预登记协议](workflow-prefix-control-v1-protocol.md)、[执行器](../src/my_project_1/workflow_prefix_control.py)、[单测](../tests/test_workflow_prefix_control.py)。

## 结论

**通过固定领域线索、完整人工流程前缀的有限工程准入。** 不证明独立业务语义构念成立，不证明模型理解、遗忘或自然流程中存在相同效应。没有加载第五checkpoint做新forward；只读流式检查权重hash。未prepare真实实验，实际outputs/workflow-prefix-control-v1-20261006不存在。

协议固定2个范围问题、3领域、2措辞，12主体与2重复控制。完整句前缀不包含下一操作对象答案；不是上一轮未完成To delete/remove cloze。review/check领域信息是配平的提示，不精确复现旧Billing/Invoices导航，所以不得把未来结果称为已验证旧错误路线的原因。neutral仍包含登录准备步骤，整单remove措辞自然性有限；两种说法只提供内部复核，非独立业务样本。

## 编码与执行契约

| 项目 | 验收证据 |
| --- | --- |
| 问题配对 | 8token，两范围只在零基5/6不同；prompt均11token |
| 前缀配平 | 共用25token，换行+route13token，共38；每组领域仅route零基5变化 |
| 实际预测位置 | 输入均49token，首次预测input_length=49；历史+回复总320，续写均余282 |
| 拼接与完整边界 | 整体重编码=共用段+route；句末完整，无强制独立空白；3种代表续写整段编码契约通过 |
| 无答案泄露 | 人工前缀不含selected/entire/remove/cancel/delete；不含候选或判读标准 |
| greedy/角色/EOS | 合成EOS oracle让角色logit占优，仍被屏蔽；独立计算EOS概率吻合 |
| 输入与记录 | 捕获合成forward实际IDs，逐条等于问题+完整前缀；记录token、stop、input_length、EOS/top5 |
| 状态与异常 | 无梯度，eval后模式恢复；异常、非有限、无预算拒绝；control/state/dependency失败标invalid并拒绝阶段完成 |
| 冻结/补缺 | 合成临时目录14记录完整流程；模拟第3次中断，补缺后旧bytes/mtime不变；完成重跑不加载模型 |
| 先校验后加载 | 损坏旧记录在load_model之前拒绝；准入失败不创建prepare目录；完整阶段重跑只读 |
| 语义记录 | 只允许新续写作为证据；支持other对象，不强把purchase当整单；恢复标签/非billing适用性、重复控制判读与缺项门控通过 |
| 语义汇总 | 只汇总12主体，保留两措辞与每侧标签；unavailable不算失败/成功，不作总体准确率 |
| 旧证据 | 不改opening-scope/v2协议、代码、报告或输出，不改14Excel、训练目标及配置 |

标签验收是结构/边界验收，不是自动语义金标准。下一动作/全文/recovery需人工阅读全文；示例“明确转回但不推进”不记恢复，实际误删账户不能因出现order而记匹配。风险v1.1、其它旁注v1.0，避免以修好某个病例作验收标准。没有候选评分、probe、训练入口或新Excel。

## 测试证据与运行边界

- 新模块 **30 passed，4.78秒**。
- 新模块＋开场执行器＋reply_prefix＋v2记录/执行回归：**92 passed，10.41秒**；不是重新跑全项目。
- 新测试autouse禁止torch.load；只有新建合成模型。共享NLL helper有独立常量oracle边界测试，但本实验没有真实候选NLL阶段，不将该测试作为模型结果。
- CLI check仅冻结tokenizer编码与权重hash检查，显示14jobs、38token前缀；不会加载权重、创建实际输出或执行prepare。
- prepare/verify绑定协议、准入、源码、测试、配置、tokenizer、真实环境和全部刺激；generate是另行授权的未来动作。本轮完成后只等待授权，不因方法通过自动运行。

未来执行入口：`.venv/Scripts/python.exe -X utf8 -u -m my_project_1.workflow_prefix_control`，依次prepare、generate、人工写semantic-reviews.json、freeze-review、summarize。原权重/解码不变，失败停止。不会启动第六训练。

## 机器准入receipt

```json
{
  "status": "accepted_for_complete_workflow_prefix_only",
  "date": "2026-10-06",
  "fifth_inference": false,
  "training": false,
  "tests": {"new": "30 passed in 4.78s", "targeted_regression": "92 passed in 10.41s"},
  "sha256": {
    "docs/workflow-prefix-control-v1-protocol.md": "1fd720d7424936d867b9d52d12caafdbf710ce81dd8f5881f2f239e8eb0d83a7",
    "src/my_project_1/workflow_prefix_control.py": "29487b75226ae7f3ac0ce247c059c03905caa5e35f03981f1f66b54b52133d5d",
    "tests/test_workflow_prefix_control.py": "223049f89261f88ab3a54bd512552e81cf85a786090bac02035f7d594d90bb86",
    "src/my_project_1/tokenization.py": "9af3671c45504dfba09be278f1111a97324eb92742a5907cbac55a24e4383546",
    "src/my_project_1/opening_scope_control.py": "4617f01153b4f42e3da4b35c502a647b423e7e6cc3d4826005d19255dfea944c",
    "src/my_project_1/condition_measurement_v2.py": "a1ae29881cc8a752633e7178bf3fd4a9a211ce6fa7a0d5b1aad7ad5bb135ab0f",
    "src/my_project_1/reply_prefix.py": "db77bcf126ab0e1600fac57df4db33f018118ea27712d9eb526ab9eff1805213"
  },
  "limitations": ["artificial complete prefixes", "domain-cue proxy not exact old navigation", "neutral includes login", "one synthetic question pair", "no independent business construct", "no causal forgetting claim"],
  "real_execution_registered": false
}
```
