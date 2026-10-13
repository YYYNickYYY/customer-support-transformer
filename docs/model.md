# 小型 Decoder-only 语言模型

实现位于 `src/my_project_1/model.py`，复用已有 Embedding 和 TransformerBlock。

## 完整数据流

```text
input_ids [B,T]
  → Token Embedding + 可学习位置编码 [B,T,128]
  → 4 个参数独立的 Pre-LN Transformer Block [B,T,128]
  → 最终 LayerNorm [B,T,128]
  → Linear(128,4000)，无 bias [B,T,4000]
```

B 是样本数，T 是当前 batch 长度。每个位置输出 4,000 个词表分数（logits），
用于预测下一个 token。它们还不是概率；交叉熵直接接收 logits，不先做 softmax。
输入词表与输出层不共享权重，便于独立理解。Linear/Embedding 权重用标准差
0.02 的正态分布初始化，bias 为零，PAD embedding 行为零。

4 个 Block 的结构相同，但不是重复调用同一组参数。
最终 LayerNorm 是所有 Block 之后额外的一次归一化。

## 回复监督与 loss

Dataset 已生成错开一位的 labels，模型不能再次错位。例如：

```text
input:  BOS USER 问题 ASSISTANT 回复1 回复2
labels: -100 -100 -100 回复1     回复2 EOS
```

问题位置虽然不直接计算 loss，仍参与上下文计算并可收到回复 loss 的梯度。
有效 labels 的交叉熵按 token 求平均，问题和 padding 的 -100 被忽略。
只有实际保留的 EOS 参与监督；截断样本不会补造 EOS。
全无监督的 batch 会报错，避免得到 NaN loss。
padding logits 置零仅为了清晰诊断，不应在这些位置采样。

```python
model = TinyDecoderLM.from_config(config)
result = model(**batch)
logits = result["logits"]
loss = result["loss"]
loss.backward()
```

不传 labels 时 loss 为 None；attention_mask 可省略，默认由 PAD ID 推导。
显式 mask 仅支持右侧 padding。

## VS Code 终端验证

无需输入命令的方式：在 VS Code 打开整个项目文件夹，点击左侧“运行和调试”，
在顶部下拉框选择“完整模型检查（CPU）”，点击绿色三角运行。
F5 为调试运行（可设置断点），Ctrl+F5 为不调试运行。
输出显示在“调试控制台”，不会要求手动激活虚拟环境。
配置位于 `.vscode/launch.json`，固定使用项目 `.venv`、项目根目录和模块运行方式。
不要用编辑器右上角“运行 Python 文件”直接运行带相对导入的模块。

下拉框还提供自动设备选择、Dataset、Block 和全部测试。
模型检查不是正式训练。

若 F5 卡在环境识别阶段，可直接按 Ctrl+Shift+B，或点击“终端 → 运行生成任务”。
这会运行默认任务“完整模型检查（无需调试器）”，结果在终端而不是调试控制台。
任务直接启动项目 Python，无需输入命令、激活环境或等待 Python 调试扩展。
工作区已关闭新版 Python 环境扩展集成，避开当前环境版本识别错误；
修改设置后需在命令面板执行“Developer: Reload Window（重新加载窗口）”。

也可继续在终端运行：

```powershell
python -m my_project_1.model --device cpu
python -m pytest tests/test_model.py -q
```

验证命令核对冻结 tokenizer revision 和特殊 ID，只读取 train 首批，执行前向
与反向，不更新参数、不保存 checkpoint，不读取 sealed_test。
随机初始化 loss 不是训练效果或质量评估。

下一步将模型接入训练循环：先少量真实样本过拟合，再实现 dev 评估和 checkpoint。
当前测试中的合成样本过拟合仅验证模型能学习，不代表客服生成能力。
