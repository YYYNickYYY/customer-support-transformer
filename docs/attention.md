# Embedding、位置编码和因果自注意力

代码：`src/my_project_1/attention.py`。在 VS Code 的项目虚拟环境中运行：

```powershell
python -m my_project_1.attention
python -m pytest tests/test_attention.py -q
```

## 从 ID 到向量

`TokenPositionEmbedding` 接收 `[B,T]` token ID，输出 `[B,T,128]`。
`nn.Embedding(4000,128)` 是可学习的表，每个 token ID 查出一行 128 个数。
`LearnedPositionEmbedding` 用位置 0..T-1 查另一张表，让相同 token 出现在不同位置时拥有不同表示。
两者相加后应用 dropout；padding 位置的结果置零。位置容量固定为 512。
这里选择可学习绝对位置编码，便于观察参数更新；后续可以再比较正弦位置编码或 RoPE。

## 一次注意力计算

1. 输入 `x` 的形状 `[B,T,128]`，通过三个独立线性层得到 Q、K、V。
2. 拆成 4 个头，变为 `[B,4,T,32]`。
3. `Q @ K.transpose(-2,-1) / sqrt(32)` 得到 `[B,4,T,T]` 分数。
4. 因果 mask 允许 key 位置 <= query 位置；padding mask 排除补齐位置。
5. 屏蔽分数设为负无穷，softmax 后未来位置和 padding 的权重为零。
6. 权重乘 V 汇总上下文，拼回 `[B,T,128]`，再经过输出线性层和 dropout。

全 padding 的 query 会特殊处理，避免 softmax 全负无穷产生 NaN。
padding query 的权重和最终输出都为零。当前约定只支持右 padding。
`return_attention=True` 可返回 dropout 前的权重，便于观察和测试；真实 query 的权重和为 1。
训练 dropout 之后的权重不要求和为 1；`eval()` 关闭 dropout。

## 与 Dataset 连接

```python
from my_project_1.attention import TokenPositionEmbedding, CausalSelfAttention

embedding = TokenPositionEmbedding(vocab_size=4000, d_model=128, max_length=512)
attention = CausalSelfAttention(d_model=128, n_heads=4, max_length=512)
x = embedding(batch['input_ids'], batch['attention_mask'])
context = attention(x, batch['attention_mask'])
```

这些对象是模型子模块；以后由完整模型统一调用 `.to(device)`。
目前输出仍是 128 维表示，尚未输出 4,000 类 logits，也尚未计算训练 loss。
下一步在注意力外添加 LayerNorm、残差及 FFN，组装 Transformer Block，最后堆叠 4 层并添加词表输出层。

## 已验证的行为

- 单头设置为单位投影，输出与手算的缩放点积注意力一致。
- 修改未来 token 不会改变前面位置的输出。
- 更改 padding 内容不影响真实 token，补齐前后的真实输出一致。
- 全 padding 样本无 NaN，padding 的梯度为零。
- token/position/Q/K/V/output 参数参与反向传播。
