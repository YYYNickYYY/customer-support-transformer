# 90 天实验日志

- 起始日期：2026-09-02
- 记录方式：所有实验按 Day 顺序追加到本文档。
- 记录原则：即使实验失败，也保留假设、证据、决策和下一步。
- 状态标记：`计划`、`进行中`、`完成`、`阻塞`、`未记录`。

## 索引

| Day | 日期 | 状态 | 主题 |
| --- | --- | --- | --- |
| [Day 01](#day-01--2026-09-02) | 2026-09-02 | 完成 | 可重建环境、工单 Schema、数据处理基线 |
| Day 02–05 | 2026-09-03 至 2026-09-06 | 未记录 | — |
| [Day 06](#day-06--2026-09-07) | 2026-09-07 | 完成 | 固定 revision 的 Bitext 数据审计 |
| Day 07 | 2026-09-08 | 未记录 | — |
| [Day 08](#day-08--2026-09-09) | 2026-09-09 | 完成 | 统一 CPU/CUDA 选择 |
| Day 09–90 | 待进行 | 计划 | 按日补充 |

---

## Day 01 — 2026-09-02

### 今日目标

1. 建立隔离且可重建的 Python 环境。
2. 起草客服工单 JSON Schema。
3. 将 Bitext 风格数据处理写成纯函数并添加测试。

### 假设

- Bitext 样本字段按 `flags`、`instruction`、`category`、`intent`、`response` 接入。
- 数据集未提供时间字段，因此 `created_at` 和 `updated_at` 保留为 `null`，不伪造时间。
- Day 1 不下载完整数据集；使用具有相同字段结构的最小测试样例。

### 实验与命令

```powershell
.\.tools\uv\uv.exe sync --extra dev
.\.venv\Scripts\python.exe -m pytest
```

### 结果与证据

- 工单结构：`schemas/work-order.schema.json`
- 数据处理：`src/my_project_1/data_processing.py`
- 自动测试：`tests/test_data_processing.py`、`tests/test_schema.py`
- 环境：项目内 CPython 3.12.14、uv 0.12.9、`.venv`、`uv.lock`。
- 测试结果：8 项测试全部通过。

### 决策

- 使用 `src/` 包布局。
- 使用 Python 3.12.14、项目内 `.venv` 和 `uv.lock`。
- 默认开发环境保持轻量，模型依赖放入 `ml` extra。
- 工单 ID 从语义字段生成稳定 SHA-256 摘要，保证重复运行结果一致。

### 问题

- 工单优先级、情绪和 PII 状态暂时无法从 Bitext 原字段可靠推断。

### 下一步

- 抽样真实 Bitext 数据，对类别、意图和 flags 的实际取值做数据剖析。

---

## Day 06 — 2026-09-07

### 今日目标

从固定 revision 的实际 Bitext CSV 统计类别、意图和 instruction 长度，不采用上游数据卡中的汇总数字。

### 实验与证据

- Revision：`fb86d1b60038970d79fefbfa0c28b47c22d8b961`
- 文件大小：19,202,474 bytes
- SHA-256：`6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434`
- 实际行数：26,872
- 实际唯一值：11 categories、27 intents
- Instruction 字符长度：最小 6、最大 92、均值 46.890、中位数 48、P95 61

完整统计见 `artifacts/bitext-profile-fb86d1b.json`。

### 决策

- 以固定 revision 实际文件的 `11/27` 为准。
- 上游 Viewer 的 `11/27` 与实际文件一致；卡片正文的 `10/27` 记录为文档漂移。
- 所有后续数据报告必须记录 revision、文件名和 SHA-256。

### 下一步

- 在训练/验证拆分前检查重复 instruction、类别分布及潜在模板泄漏。

---

## Day 08 — 2026-09-09

### 今日目标

实现 `src/utils/device.py`，为实验提供统一的 CPU/CUDA 选择入口。

### 实现与决策

- `from utils.device import get_device`；返回 `cpu` 或 `cuda` 字符串，可用于 PyTorch `.to()`。
- 默认 `auto` 优先选择可用的 CUDA，否则返回 CPU；显式指定不可用的 CUDA 时抛出异常。
- PyTorch 按需导入，未安装时自动模式返回 CPU；安装损坏或探测异常继续向调用方报告。

### 结果与证据

- 执行 `.\.venv\Scripts\python.exe -m pytest -q`：22 项测试通过。
- 新增 14 个设备选择测试用例，使用模拟对象覆盖不同硬件和依赖状态；未执行真实 GPU 计算。

---

### 追加实验：训练脚本骨架

- 新增 `src/my_project_1/train.py` 和 `configs/train.json`，复用 `get_device`。
- 每次运行独立保存 `config.json`、`loss.jsonl` 和 `status.json`。
- 真实训练通过 `train_step` 回调接入；dry-run 数据明确标记为示例。
- 全部 32 项测试通过，覆盖配置、逐步 loss、目录隔离、异常及中断保留记录等。
- 命令行 dry-run 成功生成 10 条示例 loss；本次未执行真实模型训练。

## Day 10 — 2026-09-11

### 模板组切分与测试 ID 冻结

- 按用户确认规则移除 instruction 实体占位符、统一小写和空白；保留标点。
- 使用既定 revision 和 SHA-256 校验过的 CSV，seed=42，按模板组数量 80/10/10 分配。
- 生成 `data/splits/bitext-v1/`：train 21,528 条，dev 2,688 条，sealed_test 2,656 条。
- `sealed_test_ids.json` 冻结测试 ID；`manifest.json` 记录配置、计数及文件哈希，重复执行拒绝覆盖。
- 36 项测试通过；实际输出核对通过：26,872 个唯一样本 ID、模板零交叉、冻结 ID 和哈希一致。
- 近义改写不在此次分组规则覆盖范围内；sealed_test 留给最终评估。

## Day 31 — 2026-10-02

### 训练和冻结 tokenizer，确定输入长度

- 只用 train 的 instruction/response 训练 byte-level BPE，tokenizers 0.23.1，词表 4,000。
- 冻结特殊 ID、对话模板和 tokenizer SHA-256 revision：`43b80bbb10695d80aa442fa8435eb3ca8c33111554bf1d076006dcf2b7d76ea7`。
- train/dev 输入哈希验证通过；sealed_test 未读取。
- 比较 128/256/512/1024：选择 512，完整序列截断率 train 0.0139%（3 条）、dev 0%。
- 保存 `artifacts/tokenizer/bitext-bpe-v1/`、`configs/model.json` 和 `docs/tokenizer.md`。
- 固定只对回复和保留 EOS 计算 loss；验证回复首 token 的错位标签与截断行为。
- 41 项测试通过。下一步实现 Dataset/collator 和小型因果 Transformer。

### 追加：项目整理与 GitHub 协作准备

- 将损坏 CSV、Python/pytest 缓存和过时 requirements-lock.txt 移至 `.tools/cleanup-backup-2026-10-02/`，可恢复且不上传。
- 删除空测试占位文件；保留冻结数据、tokenizer、有效原始数据缓存和运行环境。
- 新增 VS Code 配置、PR 模板、GitHub Actions 测试和冻结文件哈希校验。
- 初始化 main 分支；GitHub 上传状态以远程仓库为准。
- bootstrap 默认安装 tokenizer extra 并遵循 uv.lock，确保新环境可运行全部测试。
- 本地 41 项测试通过。

## 后续日志模板


```markdown
## Day NN — YYYY-MM-DD

### 今日目标

### 假设

### 实验与命令

### 结果与证据

### 决策

### 问题

### 下一步
```
