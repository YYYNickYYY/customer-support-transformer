# Bitext Customer Support 数据卡（项目快照）

Token 长度及截断率见 [冻结 tokenizer 报告](tokenizer.md)：train-only BPE，词表 4,000，固定模板及 max_length=512，统计范围 train/dev。

本数据卡只报告固定 revision 实际 CSV 文件的统计值。上游数据卡正文中的类别数不作为统计输入。

## 数据来源与可复现性

| 项目 | 值 |
| --- | --- |
| Hugging Face 仓库 | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` |
| 固定 revision | `fb86d1b60038970d79fefbfa0c28b47c22d8b961` |
| 文件 | `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv` |
| 文件大小 | 19,202,474 bytes |
| SHA-256 | `6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434` |
| 统计日期 | 2026-09-07 |
| 行数 | 26,872 |
| 列 | `flags`, `instruction`, `category`, `intent`, `response` |

固定文件链接：[Hugging Face revision 文件](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/blob/fb86d1b60038970d79fefbfa0c28b47c22d8b961/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv)。

统计脚本为 `scripts/profile_bitext.py`，完整机器可读结果为 `artifacts/bitext-profile-fb86d1b.json`。脚本会校验必要列，并可通过 `--expected-sha256` 拒绝错误文件。

## 实际文件统计

`category`、`intent` 和 `instruction` 均无缺失值。

### Category：11 个唯一值

| Category | 行数 | Intent 数 |
| --- | ---: | ---: |
| ACCOUNT | 5,986 | 6 |
| CANCEL | 950 | 1 |
| CONTACT | 1,999 | 2 |
| DELIVERY | 1,994 | 2 |
| FEEDBACK | 1,997 | 2 |
| INVOICE | 1,999 | 2 |
| ORDER | 3,988 | 4 |
| PAYMENT | 1,998 | 2 |
| REFUND | 2,992 | 3 |
| SHIPPING | 1,970 | 2 |
| SUBSCRIPTION | 999 | 1 |

唯一值为：`ACCOUNT`, `CANCEL`, `CONTACT`, `DELIVERY`, `FEEDBACK`, `INVOICE`, `ORDER`, `PAYMENT`, `REFUND`, `SHIPPING`, `SUBSCRIPTION`。

### Intent：27 个唯一值

- ACCOUNT：`create_account` (997), `delete_account` (995), `edit_account` (1,000), `recover_password` (995), `registration_problems` (999), `switch_account` (1,000)
- CANCEL：`check_cancellation_fee` (950)
- CONTACT：`contact_customer_service` (1,000), `contact_human_agent` (999)
- DELIVERY：`delivery_options` (995), `delivery_period` (999)
- FEEDBACK：`complaint` (1,000), `review` (997)
- INVOICE：`check_invoice` (1,000), `get_invoice` (999)
- ORDER：`cancel_order` (998), `change_order` (997), `place_order` (998), `track_order` (995)
- PAYMENT：`check_payment_methods` (999), `payment_issue` (999)
- REFUND：`check_refund_policy` (997), `get_refund` (997), `track_refund` (998)
- SHIPPING：`change_shipping_address` (973), `set_up_shipping_address` (997)
- SUBSCRIPTION：`newsletter_subscription` (999)

### Instruction 长度

| 口径 | 最小 | 最大 | 均值 | 中位数 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unicode 字符数 | 6 | 92 | 46.890 | 48 | 61 |
| 非空白词段数 | 1 | 16 | 8.691 | 9 | 13 |

字符数定义为 CSV 解码后直接执行 `len(instruction)`，不做 `strip` 或文本规范化；词段数定义为正则 `\S+` 的匹配数。P95 使用 nearest-rank 方法。

## 11/27 与 10/27 的差异

- 固定 revision 的实际 CSV：**11 categories / 27 intents**。
- Hugging Face Dataset Viewer：**11 / 27**，与实际文件统计一致。
- 上游数据卡正文曾声明：**“27 intents assigned to 10 categories”**。

因此，本项目将 `11/27` 视为该 revision 的事实值，并把 `10/27` 标记为上游文档与数据之间的漂移。尤其是实际文件包含 `CONTACT` 类别及其两个 intent；实际类别命名还使用 `CANCEL`、`SHIPPING`、`SUBSCRIPTION`，而不是仅依赖正文中的分类描述。后续统计必须同时记录完整 revision 与文件哈希，不能只记录 `main`。

## 复现命令

## 固定模板组切分（2026-09-11）

输出目录：`data/splits/bitext-v1/`。保留原始五列，追加 `id`、`template_id`、`source_row`。
模板规则：将 `instruction` 中的 `{{...}}` 实体占位符替换为空格，调用 `lower()`，合并空白并去除首尾空白。保留标点，不做语义改写或近似匹配。
只根据 instruction 分组，不使用类别、意图或回复。同一规范化字符串始终处于同一个集合。

| 集合 | 样本数 | 模板组数 |
| --- | ---: | ---: |
| train | 21,528 | 19,192 |
| dev | 2,688 | 2,399 |
| sealed_test | 2,656 | 2,399 |

固定 seed=42；将排序后的模板键随机打乱，按组数量 80/10/10 切分（边界向下取整），不是按样本数量分配，也未按 intent 分层。
固定源 revision 和 SHA-256 沿用本数据卡上方记录，输入哈希不匹配会中止。
本次确认 26,872 行完整且仅出现一次，三个集合间模板交叉为零。

`sealed_test_ids.json` 冻结 2,656 个测试 ID 和模板 ID；ID 定义为 `bitext:<完整revision>:<从0开始的CSV记录索引>`。
`manifest.json` 保存切分规则、数量与各输出文件 SHA-256。脚本拒绝覆盖已有输出目录。
应将冻结 ID 与 manifest 一起纳入版本管理，后续实验复用此划分；sealed_test 仅用于最终评估，不参与调参或模型选择。
这里的“冻结”通过清单及覆盖保护实现，不是文件权限锁；“sealed”是使用约定，不是加密。
该规则防止规范化后完全一致的模板交叉，但不保证语义相似句或标点变体没有泄漏。

```powershell
.\.venv\Scripts\python.exe -m my_project_1.split_data <固定CSV路径> --output data/splits/bitext-v1 --seed 42
```

### 统计复现

下载固定 revision 的 CSV 并验证 SHA-256 后运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\profile_bitext.py `
  <CSV路径> `
  --revision fb86d1b60038970d79fefbfa0c28b47c22d8b961 `
  --expected-sha256 6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434 `
  --output .\artifacts\bitext-profile-fb86d1b.json
```
