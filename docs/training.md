# 全量 train 训练与 dev 评估

入口 `src/my_project_1/training.py`，参数 `configs/training.json`。
默认从随机初始化开始，不接着 4 条样本过拟合的权重训练。
冻结 tokenizer、模板、最大长度 512 和模型结构不变；sealed_test 不被读取。

## VS Code 运行

“终端 → 运行任务 → 全量 train 训练 + dev 评估（5轮，耗时较长）”。
该任务不是默认生成任务，避免误触 Ctrl+Shift+B 就启动长时间训练。
“训练流程检查（16条 train + 16条 dev）”只验证流程，不代表全量结果。

默认参数：5 轮、batch_size=8、AdamW 学习率 0.0003、weight_decay=0.01、
梯度裁剪 1、seed=42。自动选 CUDA 或 CPU；CPU 线程数 2。
当前 CPU 环境下全量训练可能较久，不提供未经实测的时间承诺。
显存不足时可减小 batch_size，不要通过改变冻结 tokenizer 或上下文长度解决。

train 共 21,528 条，每轮随机打乱，完整遍历 2,691 个 batch，不丢末批；
dev 共 2,688 条，每次完整遍历 336 个 batch，不打乱。
没有设 train_limit/dev_limit 才是全量；流程检查运行明确标记 subset_run=true。

## 训练与评估规则

先记录初始 dev loss，然后每轮 train 前向、反向、梯度裁剪、AdamW 更新。
标签已错位，只监督回复及保留 EOS，忽略问题和 padding 的 -100。
每轮完成后用 eval/no_grad 评估 dev；按所有有效 token 的 NLL 求平均，
不能简单平均各 batch loss，否则短 batch 和长 batch 权重不正确。
训练 epoch loss 同样按监督 token 汇总，但它来自参数不断更新且启用 dropout 的过程，
并非 epoch 结束时对 train 的独立评估，因此与 dev loss 有这两项区别。

在完成的训练轮次中按最低 dev loss 保存 best.pt；best_epoch 可能不是最后一轮。
epoch0 只作随机初始化基线，不参加最佳训练模型选择。
dev 用于选择模型，不等于最终独立测试；sealed_test 仍封存。

## 每次运行的文件

唯一目录 `runs/<时间>-full-<ID>/`：

- config.json / experiment.json：配置快照、设备、train/dev 数量和冻结哈希。
- loss.jsonl：每个 batch 的真实训练 loss、样本数和监督 token 数。
- metrics.jsonl：初始 dev 与每轮 train/dev 指标，最佳轮次。
- best.pt：最低 dev loss 的模型及优化器状态。
- last.pt：每 200 步和每轮结束保存；写入临时文件后替换，避免破坏旧 checkpoint。
- status.json / summary.json：运行状态和完整成功时的指标摘要。

中断保留已写入的日志和既有 checkpoint；少于 checkpoint_every 的早期中断可能尚无模型文件。
不会删除旧运行。目前不提供断点续训，重新运行会从头开始；优化器状态保存用于后续扩展。
不要把 smoke-test 的 best.pt 当作全量训练模型。
运行文件被 Git 忽略。新模型用于生成时，需要在 generation.json 中明确更新 checkpoint 路径；
当前已手动切换为全量训练第三轮 best.pt；以后新训练仍不会自动替换生成配置。

## 第三次：仅处理结果/隐私保证目标干预（2026-10-05）

使用 `configs/training-third.json`，原 `configs/training.json` 不变。
VS Code“运行任务”中新加“第三次训练（冻结保证话术干预，5轮）”；该任务仅供需要开始新运行时使用。
本次已由助手启动后台全量训练，**不要再点击该任务重复训练**。
当前实时输出在 `outputs/third-training-20261005/stdout.log`，错误输出在同目录 `stderr.log`。

规则见 [冻结干预协议](train-target-revision-v1.md)。完整衍生数据及260条修改清单位于
`data/derived/result-privacy-v1/`，冻结 manifest 保存哈希、规则与源文件依赖。
Dataset 先核对原train的冻结哈希，再核对衍生文件并重放精确规则；不关闭原数据校验，也不重训tokenizer。
训练的 experiment.json 和 checkpoint 记录train_revision及实际train哈希，避免把原数据哈希误称为实际训练目标哈希。

train仍21528、dev仍2688；5轮、lr0.0003、batch8、seed42、模型和生成配置不变。
从新初始化开始，保存单独best/last和loss记录，不覆盖此前两次模型。
保守规则改变260条（1.21%），不代表已经清除所有无依据断言；支付/配送/账户流程与实体、风格问题保留。
训练完成后需另行将固定dev生成的checkpoint切至第三次best.pt（其他生成参数不变），不能提前自动切换。

## 原流程检查记录

16 train / 16 dev、1 轮、2 次更新：初始 dev loss 8.350767，最终 8.005147。
这是流程验证，不是完整数据集成绩；正式运行必须使用默认无 subset 限制的配置。
