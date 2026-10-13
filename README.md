# my-project-1

[![Tests](https://github.com/YYYNickYYY/my-project-1/actions/workflows/tests.yml/badge.svg)](https://github.com/YYYNickYYY/my-project-1/actions/workflows/tests.yml)

从零实现小型Decoder-only Transformer的90天学习项目：冻结数据与tokenizer、回复监督Dataset、因果注意力、Transformer Block、训练/dev评估、自回归生成及单因素实验。

本轮优化已收尾，推荐第七次6层、5轮模型作为本地学习版本。第八次8轮降低dev loss，但同135条greedy回复未明确整体改善，因此不替换。不是可部署客服系统，不继续逐条修补。

## 目标与已完成的工程闭环

目标不是调用预训练大模型获得最佳客服效果，而是亲手实现并理解一个小模型从数据到回复的完整过程。模型使用PyTorch构建，不是对现成GPT做微调。

- **数据**：固定Bitext revision，按去实体占位符、统一大小写/空白后的instruction模板分组划分train/dev/sealed_test，冻结测试ID；从实际文件统计11类category、27类intent。
- **输入与监督**：仅在train上训练并冻结4000词表BPE；最大长度512，动态padding、attention mask、错位标签；只监督回复及保留EOS，问题/padding不计loss。
- **模型**：Embedding与位置编码 → 因果多头注意力/残差/LayerNorm/FFN组成的Block → 最终LayerNorm → 词表输出。实现4层与6层配置；6层版本约228万参数，d_model128、4头、FFN512。
- **训练**：少样本过拟合验证、全量train、token加权dev loss、AdamW、梯度裁剪、配置/loss日志、best/last checkpoint和中断状态。当前未实现完整断点续训。
- **生成与评估**：原greedy和束搜索学习实现；固定135条dev逐回复评估意图、帮助、通顺、重复、无依据风险与实体。保留输入token、停止原因、哈希和证据边界。
- **决策**：依次检验数据话术、解码、学习率、容量和训练预算；没有整体收益的方案不采用，本轮有明确停止点。

代码、单测和实验记录共同构成成果；承认哪些实验无收益也是结论，不以局部好例子代替整体结果。

## 实验主线与最终结果

前两次建立训练基线并比较3→5轮；第三至第五次在其他设置固定时审查、改写处理结果/隐私/退款承诺目标，并统一评分边界。中间完成生成诊断和解码对照，未把每次诊断都变成新训练。最后比较学习率、模型深度和轮数：

| 实验 | 唯一主要变化/角色 | best dev loss ↓ | 意图/2 ↑ | 帮助/2 ↑ | 重复条数 ↓ | 无依据风险条数 ↓ | 决策 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 第五 | 后续实验基线，4层/5轮/LR 0.0003 | 1.1029 | 1.8074 | 1.3259 | 63 | 43 | 保留历史基线 |
| 第六 | 对第五仅LR 0.0003→0.0001 | 1.3272 | 1.6593 | 1.0889 | 61 | 31 | 不采用 |
| 第七 | 对第五仅4→6层 | 1.0663 | 1.8593 | 1.4000 | 45 | 41 | 当前学习版本 |
| 第八 | 对第七仅5→8轮 | 0.9818 | 1.8889 | 1.3926 | 51 | 40 | 不替换第七 |

生成评分均基于同135条问题、原greedy，风险v1.1/其他维度v1.0及校准记录；dev loss覆盖全部2688条dev，不等于这135条的任务成功率。第七使用最终校准分而非初稿。详见[第六](docs/sixth-evaluation-20261007.md)、[第七](docs/seventh-evaluation-20261007.md)、[第八](docs/eighth-evaluation-20261013.md)报告。

第八loss约降7.93%，帮助却未提升、重复增加，说明**更好地预测参考答案的下一个token，不保证沿着自身历史生成的整段回复更有用**。例如修改商品的澄清有所改善，但也出现“编辑账户→删除账户”的退步，因此保留第七而不是机械采用最低loss版本。

这些是单seed、反复使用的开发集、非盲AI代评结果，不宣称统计显著性、独立测试泛化或真实客服业务能力。sealed_test没有用于本轮模型选择，尚未进行最终独立测试。

## 阅读入口

- [唯一实验日志](experiments/experiment-log.md)：实验过程、最终决策和“loss改善为什么不保证生成更好”的学习结论。
- [项目导航](项目导航.md)：当前入口、历史资料与本地文件说明。
- [第八次比较报告](docs/eighth-evaluation-20261013.md)：最终配对结果；原始评分/权重保留本地。
- [VS Code与GitHub PR](docs/github-workflow.md)：协作流程。

## 重建环境与测试

Python 3.12.14、uv 0.12.9和`uv.lock`固定版本。Windows PowerShell：

```powershell
git clone https://github.com/YYYNickYYY/my-project-1.git
cd my-project-1
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -WithData
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_artifacts.py
```

已有uv时，也可以使用：

```powershell
uv sync --locked --extra dev --extra data
uv run --no-sync python -m pytest -q
```

`data`额外依赖包含本项目所需的PyTorch、NumPy和tokenizers；只有需要额外的transformers库时才使用`-WithMl`。激活虚拟环境不是必须的，直接调用`.venv`中的Python即可。

完整本地实验回归还需要未上传的运行记录、评审输出和个人工作台。源码检出保留模型/数据/生成单测，只对明确缺少本地产物的实验回放检查标记skip；它不能冒充完整历史实验验收。已存在的本地产物若损坏，仍应失败，不以跳过掩盖。

### 最短可运行演示：不训练、不需要历史权重

```powershell
.\.venv\Scripts\python.exe -m my_project_1.model --config configs/model-six-layer.json --device cpu
```

该命令用冻结tokenizer读取4条真实train样本，执行随机初始化6层模型的前向/反向诊断；没有optimizer更新或checkpoint保存。源码副本中已验证：

```text
Input: (4, 279)
Logits: (4, 279, 4000)
Layers: 6
Parameters: 2279424
Supervised tokens: 584
All gradients finite: True
Diagnostic only: no optimizer update or checkpoint has been made.
```

这证明数据到模型的管线可以运行，不代表随机模型会生成有效回复。

### 可选：自己运行公开基线训练与生成

以下命令会真正更新参数，仅供读者主动选择；本次发布未执行新训练：

```powershell
# 原始数据、4层基线的16条train/16条dev流程检查，不是第七次复现
.\.venv\Scripts\python.exe -m my_project_1.training --config configs/training.json --smoke-test
# 去掉 --smoke-test 才是该配置的完整5轮训练
```

新运行会在`runs/`独立保存配置、loss、dev指标、best/last权重，不覆盖旧结果。要对自己训练的模型生成，可在本地创建`configs/generation-local.json`（不提交），将`checkpoint`改成新运行实际路径，其余采用以下原greedy设置：

```json
{
  "checkpoint": "runs/REPLACE_WITH_YOUR_RUN/best.pt",
  "device": "cpu",
  "cpu_threads": 2,
  "max_new_tokens": 320
}
```

```powershell
.\.venv\Scripts\python.exe -m my_project_1.generate --config configs/generation-local.json --instruction "I need to remove one item from my order."
```

`REPLACE_WITH_YOUR_RUN`必须替换；16条流程检查模型的回复不代表全量训练效果。这条公开基线管线不依赖历史评审产物，但数据/模型配置不同于第七次，不能把其结果称为复现第七次。历史单因素实验入口保留严格证据门控，完整复放还需要本地`runs/`、`outputs/`等依赖。

## 文件布局与上传边界

| 路径 | 内容 | 是否上传 |
| --- | --- | --- |
| `src/my_project_1/`、`src/utils/` | 数据、模型、训练、生成及设备代码 | 是 |
| `tests/` | 单测与冻结证据验收 | 是 |
| `configs/` | 模型及历史实验配置 | 是 |
| `scripts/` | 重建、验证、历史审计与评分汇总工具 | 是 |
| `docs/`、`experiments/` | 方法文档、冻结协议、报告、唯一日志 | 是 |
| `data/`、`artifacts/`、`schemas/` | 已冻结数据及修订、tokenizer、清单、Schema | 是；保持原始字节 |
| `.github/`、`.vscode/` | PR检查及通用编辑器配置/历史任务 | 是 |
| `*.code-workspace` | 个人项目/实验工作台 | 否；本地保留 |
| `runs/`、`outputs/` | 权重、loss、回复、评分及历史工作簿 | 否；本地保留 |
| `.venv/`、`.python/`、`.tools/`、各类缓存 | 本地环境、工具与下载缓存 | 否 |

数据来源与许可见[DATA_LICENSE.md](DATA_LICENSE.md)，固定revision统计见[数据卡](docs/data-card.md)。训练与生成不使用sealed_test；其ID继续冻结，发布完整性哈希检查不属于模型评估。

## 运行边界

- [模型实现](docs/model.md)、[训练实现](docs/training.md)、[生成实现](docs/generation.md)说明完整管线。
- `my_project_1.train`与`configs/train.json`是早期训练骨架，dry-run使用示例loss；真实全量训练在`my_project_1.training`及各实验入口中。
- 历史实验入口会校验本地证据，只有源码时不可直接复现历史运行。不要删除门控或改写冻结哈希来强行通过。
- 仓库不含训练权重。生成配置中的历史checkpoint路径不会在新电脑自动存在；本次未改成下载权重或自动训练。
- VS Code历史任务仍可能启动训练，尤其默认构建任务；查看结果应打开报告，不要重新运行训练。本轮不启动任何新的优化实验。
