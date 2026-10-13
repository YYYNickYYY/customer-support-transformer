# Dataset 与动态 batch padding

在 VS Code 打开项目并选择 `.venv/Scripts/python.exe` 为解释器。
项目环境已有 PyTorch；换机器重建时运行 `scripts/bootstrap.ps1 -WithData`。
不传 WithData/WithMl 的基础重建会移除未选中的 PyTorch 依赖。

```python
from torch.utils.data import DataLoader
from my_project_1.dataset import CustomerSupportDataset, DynamicPaddingCollator

train = CustomerSupportDataset('train')
dev = CustomerSupportDataset('dev')
train_loader = DataLoader(
    train, batch_size=8, shuffle=True,
    collate_fn=DynamicPaddingCollator(train.pad_token_id),
)
dev_loader = DataLoader(
    dev, batch_size=8, shuffle=False,
    collate_fn=DynamicPaddingCollator(dev.pad_token_id),
)
batch = next(iter(train_loader))
```

Dataset 校验冻结 tokenizer 及所选 split 的哈希，保持 train/dev 绑定关系。
原始行存入内存，按需编码。默认封存测试集不通过此训练入口加载。
`prepare_example` 负责插入控制 token、右截断、输入/标签错位和回复监督。
max_length=512 计完整序列，错位后的输入最长 511。

collator 返回三个 CPU int64 张量，形状均为 `[batch_size, 当前batch最长输入长度]`：

| 字段 | 真实位置 | 右侧 padding |
| --- | --- | --- |
| input_ids | 输入 token ID | PAD=0 |
| labels | 回复/EOS 目标，其余 -100 | -100 |
| attention_mask | 1 | 0 |

attention_mask 只是 padding mask，未来的 Transformer 还必须实现 causal mask。
labels 已经错位，训练时直接与 logits 对齐，不再做第二次 shift：

```python
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    batch['labels'].reshape(-1),
    ignore_index=-100,
)
```

验证时按有效标签数量聚合 loss，避免不同长度 batch 的均值权重错误。
当前模块不搬运 GPU 张量；训练循环再把三个张量 `.to(device)`。
截断不伪造 EOS；没有监督 token 的样本明确报错，避免空监督 loss。

VS Code 集成终端可执行：

```powershell
.\.venv\Scripts\python.exe -m my_project_1.dataset --split train
.\.venv\Scripts\python.exe -m my_project_1.dataset --split dev
.\.venv\Scripts\python.exe -m pytest tests/test_dataset.py -q
```
