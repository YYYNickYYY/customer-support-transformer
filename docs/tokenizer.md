# 冻结 tokenizer 与输入格式

日期：2026-10-02。训练范围仅为 train 的 instruction/response，dev 只做统计，sealed_test 未读取。
源数据 revision 与 train/dev 文件哈希记录在 `artifacts/tokenizer/bitext-bpe-v1/config.json`。

## 第一版固定配置

- byte-level BPE，词表 4,000（包含特殊 token），最小频次 2，完整 256 字节基础字母表。
- tokenizers 0.23.1；保留原文大小写和空白，不进行 Unicode 归一化。
- 特殊 ID：PAD=0、BOS=1、USER=2、ASSISTANT=3、EOS=4。
- 本地 tokenizer revision（tokenizer.json 的 SHA-256）：`43b80bbb10695d80aa442fa8435eb3ca8c33111554bf1d076006dcf2b7d76ea7`。
- 冻结目录拒绝覆盖；后续使用 `load_frozen()` 验证各文件哈希。
- 模型输入配置：`configs/model.json`（模型实现尚待完成）。

固定模板：

```text
<BOS><USER>{instruction}<ASSISTANT>{response}<EOS>
```

两段文本分别编码后插入特殊 ID，避免字段边界的 BPE 合并。文本中若含保留的特殊 token 则拒绝处理。
instruction/response 单独统计时不含特殊 token；完整序列长度为两者之和加 4。

## 实际 token 长度

| Split | 字段 | 最小 | 均值 | P50 | P95 | P99 | 最大 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | instruction | 2 | 9.972 | 10 | 15 | 17 | 23 |
| train | response | 12 | 127.963 | 105 | 265 | 374 | 519 |
| train | 完整序列 | 22 | 141.935 | 119 | 279 | 389 | 533 |
| dev | instruction | 2 | 10.005 | 10 | 15 | 17 | 22 |
| dev | response | 13 | 129.885 | 106 | 266 | 380 | 474 |
| dev | 完整序列 | 24 | 143.890 | 120 | 282 | 394 | 491 |

分位数采用 nearest-rank。截断率定义为长度严格大于上限的样本数除以样本总数。

| max_length | train 完整序列截断数/率 | dev 完整序列截断数/率 |
| --- | ---: | ---: |
| 128 | 8,937 / 41.5134% | 1,168 / 43.4524% |
| 256 | 1,594 / 7.4043% | 211 / 7.8497% |
| 512 | 3 / 0.0139% | 0 / 0% |
| 1024 | 0 / 0% | 0 / 0% |

选择规则：两个 split 均不超过 1% 完整序列截断率、且无零监督样本的最小候选值。
因此固定 max_length=512，模型位置容量也设为 512。
在此上限下 instruction 单独超长率均为 0；response 单独超长为 train 1 条（0.00465%）、dev 0 条。
实际按完整序列截断时 train 丢失 44 个回复 token 和 3 个 EOS；dev 无截断。

## 训练标签约定

完整序列先向右截断至 512，然后错位：`input_ids=sequence[:-1]`，`labels=sequence[1:]`。
因此当前数据处理函数生成的模型输入最多 511 个 token；512 是完整训练序列预算，模型允许 512 个位置。
只对回复和保留下来的 EOS 计算 loss，提示部分标签为 -100；以后 batch padding 的标签也须设为 -100。
被截断的回复末尾不补造 EOS。完整序列超长也不等于全部样本无效，本次所有样本仍有回复 token 可监督。

## 命令与加载

安装轻量 tokenizer 依赖：

```powershell
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location).Path '.python'
$env:UV_CACHE_DIR = Join-Path (Get-Location).Path '.uv-cache'
.\.tools\uv\uv.exe sync --locked --extra dev --extra tokenizer
```

训练并统计（已有输出目录会拒绝覆盖）：

```powershell
.\.venv\Scripts\python.exe -m my_project_1.tokenization
```

后续训练加载已冻结的版本：

```python
from my_project_1.tokenization import load_frozen, prepare_example

tokenizer, config = load_frozen('artifacts/tokenizer/bitext-bpe-v1')
sample = prepare_example(tokenizer, 'Where is my order?', 'Please provide your order number.', config['max_length'])
```

`freeze.json` 保存 tokenizer、配置和统计结果的 SHA-256。重新训练可能因 BPE 并列频次等实现因素产生不同哈希，应视为新 revision；精确复现编码应加载冻结文件。
再次运行基础 bootstrap 会同步基础依赖；使用 tokenizer 前需保留上述 tokenizer extra（或安装包含它的 ML 依赖）。
