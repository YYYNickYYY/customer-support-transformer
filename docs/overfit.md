# 少量真实样本训练

这是验证训练链路的过拟合实验，不是验证集评估或客服生成质量评估。
只取冻结 train 文件的前 4 条，重复训练 120 步；不读取 dev/sealed_test。
冻结 tokenizer、模板、上下文预算和模型结构保持不变。

## 不输入命令的运行方式

在 VS Code 按 **Ctrl+Shift+B**，默认任务已改为
“少量真实样本训练（CPU，120步）”。每 10 步在终端显示 loss。
也可用“终端 → 运行任务”选择该任务，或原来的完整模型检查任务。
每次运行从随机初始化开始，创建新目录，不覆盖旧结果，也不从旧 checkpoint 续训。

参数在 `configs/overfit.json`：4 条样本、120 步、AdamW 学习率 0.003、
weight_decay=0.01、梯度裁剪阈值 1、seed=42、CPU 线程数 2。
模型 dropout=0.1，训练时启用；比较初始/最终 loss 时关闭 dropout，
在完全相同的 batch 上评估，使对比更公平。固定种子便于重复实验，
但不同设备/软件环境不保证逐位一致。

## 每步发生什么

1. `model.train()`，清空上一步梯度。
2. 输入固定 batch，计算回复及保留 EOS 的交叉熵；labels 已错位，不再次错位。
3. `loss.backward()` 计算梯度。
4. 检查并裁剪梯度，`optimizer.step()` 真正更新参数。
5. 将更新前的训练 loss 追加到 `loss.jsonl`。

问题和 padding 标签仍为 -100；问题表示可以通过回复 loss 获得梯度。
训练 loss 因 dropout 可以波动，不要求每一步严格下降。
“明显下降”的诊断阈值是相同 batch 的 eval loss 至少下降 50%。

## 保存的文件

每次运行创建唯一的 `runs/<run-id>/`：

- `config.json`：训练循环参数、设备与开始时间。
- `experiment.json`：完整模型/实验参数、冻结 revision、train 哈希、样本 ID 和初始指标。
- `loss.jsonl`：每步真实训练 loss，不是 dry-run 的演示值。
- `status.json`：训练循环状态与完成步数；中断时保留已有 loss。
- `summary.json`：同 batch 初始/最终 eval loss 与下降比例。
- `checkpoint.pt`：最终模型和优化器参数，当前不提供断点续训入口。

训练循环中断不会保存最终 checkpoint；status 的 completed 只表示优化循环完成，
完整成功结果还应同时具备 summary 和 checkpoint。
这些运行产物被 Git 忽略，不会自动上传 GitHub；文字记录统一写入实验日志。
只加载自己生成或可信来源的 checkpoint。

## 已完成实验

2026-10-03 的固定 4 样本实验：初始 eval loss=8.339753，最终=0.003051，
下降 99.9634%。同 batch 的 teacher-forced token accuracy=100%。
这里每个位置都能看到真实的历史 token；100% **不等于**自由生成正确率，
也不能证明新问题的泛化能力。

下一步可实现自回归生成，观察这 4 条样本的回复；之后再开展更大 train 训练与 dev 评估。
