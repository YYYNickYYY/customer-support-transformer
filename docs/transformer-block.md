# 一个完整的 Transformer Block

实现：`src/my_project_1/transformer.py`。这是 Pre-LayerNorm 的 decoder block。
输入和输出均为 `[batch, length, 128]`，每个 Block 有两条残差连接：

```python
x = x + attention(norm1(x), attention_mask)
x = x + ffn(norm2(x))
```

## LayerNorm

`nn.LayerNorm(128)` 单独处理每个 token 的 128 个特征，计算均值、方差，再通过可学习的缩放和平移调整。
它不会把不同位置或不同样本混在一起，因此不会引入未来信息。
采用 Pre-LN：先归一化副分支输入，残差主路径直接传递原表示，便于后续堆叠训练。
两次 LayerNorm 使用不同的参数，epsilon=1e-5 防止方差过小时除零。

## FFN

每个 token 使用相同的两层网络：`Linear(128,512) → GELU → Linear(512,128) → Dropout`。
它加工当前 token 的特征；不同 token 之间的信息交换由注意力负责。
最终回到 128 维才能与残差相加。注意力和 FFN 的输出分支各自负责 dropout，不对残差主路径再加 dropout。

## Padding 与梯度

先将 padding 输入置零，每次残差相加后也将 padding 置零，避免 FFN、LayerNorm 的偏置在 padding 上产生非零输出。
注意力仍合并因果和 padding mask；全 padding 样本无 NaN。
零更新分支时，Block 能完整保留真实输入，同时保留恒等梯度路径。

## 在 VS Code 验证

```powershell
python -m my_project_1.transformer
python -m pytest tests/test_transformer.py -q
```

演示从冻结 tokenizer 和真实 train batch 读取数据，通过 Embedding 和一个 Block。
演示用输出平方均值检查梯度，这只是诊断标量；当前 Block 还没有词表输出层和语言模型交叉熵。
返回注意力权重时调用 `block(x, mask, return_attention=True)`。

下一步：堆叠 4 个独立 Block，添加最终 LayerNorm 和 `Linear(128,4000)` 词表输出层，与已经错位的 labels 对齐计算交叉熵。
