# 自回归生成检查

实现位于 `src/my_project_1/generate.py`。它加载配置指定的 checkpoint，当前为全量训练第三轮的 best.pt，
只把问题作为初始输入，绝不把参考回复送给模型。

```text
输入：BOS USER 问题 ASSISTANT
预测回复 token 1 → 接回输入
预测回复 token 2 → 接回输入
…… → 遇到 EOS 停止
```

每步取最后位置的 logits，选择分数最大的 token（greedy，不随机采样），
屏蔽 PAD/BOS/USER/ASSISTANT，允许 EOS。模型处于 eval 模式、禁用梯度。
当前每步重算整个前缀，不使用 KV cache，适合小模型学习与检查。
只有模型自己预测的回复参与下一步；参考文本仅用于最后显示和比较。

## VS Code 一键运行

Ctrl+Shift+B 的默认任务现在是“生成回复（全量训练最佳模型，4条train问题）”。
原训练任务仍可通过“终端 → 运行任务”选择，不会被删除。
`configs/generation.json` 固定 checkpoint 路径、CPU、线程数 2、最多生成 320 token。
若模型文件不存在，请先训练并将配置中的 checkpoint 改为对应的新路径。

默认读取实验记录中的 4 个 train 样本 ID，逐条输出问题、生成回复、参考回复和停止原因。
不读取 dev/sealed_test，不修改 checkpoint，也不重新训练。
自定义问题可通过模块入口的 `--instruction` 参数传入；新问题输出只是观察，不能作为系统性评估。

## 长度与停止

- EOS：模型主动预测结束；EOS 计入生成 token 数，但不出现在解码文本中。
- max_new_tokens：达到生成上限，可能是未完成的回复。
- context_limit：问题和生成 token 总数达到冻结完整序列预算 512。

问题过长会报错，不悄悄截断。不会为了让输出完整而补造 EOS。
最后一次预测的输入最长为 511，保持完整序列预算和既有训练错位规则一致。

## 保存结果

每次检查在 checkpoint 同目录下创建新的 `generations/<run-id>/`，保存
`config.json`、`replies.jsonl` 和 `summary.json`。配置包含 checkpoint SHA-256、
tokenizer revision、解码方法与长度限制，便于复核。
只加载自己生成或可信来源的 checkpoint；使用 weights_only=True。

这项检查证明生成链路是否正确、模型能否复现已记住的回复，不能证明对新问题的泛化能力。

2026-10-04 实际检查：4/4 回复与训练参考文本完全一致，全部由预测 EOS 停止，
生成 token 数（含 EOS）为 46、58、265、215。参考回复没有参与生成。

随后切换全量训练第三轮 best.pt（dev loss=1.228022）重新检查：
4 条回复均由 EOS 停止，生成 token 数为 195、186、188、183，0/4 逐字匹配。
内容是相近的取消订单操作说明；前两条询问类问题没有像参考文本那样请求进一步说明，
且有“instructions or instructions”的局部重复。不能仅凭正常停止和通顺文本判定答案正确。
