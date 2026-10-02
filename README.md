# my-project-1

GitHub 协作与 VS Code PR 操作见 [工作流说明](docs/github-workflow.md)。

一个隔离、可重建的 90 天 LLM 实验项目。

## 重建环境

项目使用 Python 3.12.14、uv 0.12.9 和 `uv.lock` 固定环境与依赖版本。
首次使用或需要完全重建时，只需 PowerShell，无需预装 Python：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
python -m pytest
```

模型实验需要额外安装 ML 依赖：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -WithMl
```

## 项目结构

- `src/my_project_1/`：项目代码
- `tests/`：自动化测试
- `schemas/`：工单 JSON Schema
- `experiments/`：90 天实验日志
- `scripts/bootstrap.ps1`：从零重建项目内 Python 与虚拟环境
- `uv.lock`：当前依赖锁文件
- `src/utils/`：设备等通用工具
- `configs/`：训练和模型配置
- `data/splits/`：冻结的数据划分和测试 ID
- `artifacts/`：冻结 tokenizer 及统计报告
- `.github/`：PR 模板与自动测试
- `.vscode/`：Python 测试配置和推荐扩展

数据来源与使用条款见 [DATA_LICENSE.md](DATA_LICENSE.md)。

## 数据审计


- [Bitext 数据卡](docs/data-card.md)：固定 revision 的实际类别、意图和 instruction 长度统计
- [统计结果 JSON](artifacts/bitext-profile-fb86d1b.json)：可供程序读取的完整统计结果
- `scripts/profile_bitext.py`：从实际 CSV 重算统计并校验文件哈希

## 训练骨架

使用示例配置验证循环与保存流程：

```powershell
.\.venv\Scripts\python.exe -m my_project_1.train --config configs\train.json --dry-run
```

每次运行生成 `runs/<时间戳-唯一ID>/`，包含 `config.json`（配置、实际设备及模式）、
`loss.jsonl`（每步 epoch、step、loss 及模式）、`status.json`（完成步数及结束状态）。
`--dry-run` 的 loss 是示例数值，不代表模型训练结果。输出路径相对于执行命令的目录。

真实训练通过 `run_training(config, train_step)` 接入；`train_step(epoch, step, device)`
负责前向计算、反向传播和参数更新，返回 `loss.item()`。例如，准备好模型和数据后：

```python
from my_project_1.train import TrainConfig, run_training
from utils.device import get_device

config = TrainConfig(device="auto")
model.to(get_device(config.device))
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
model.train()

def train_step(epoch, step, device):
    # 在此获取 batch，将张量移到 device，并计算模型 loss。
    optimizer.zero_grad()
    loss = compute_batch_loss(model, epoch, step, device)
    loss.backward()
    optimizer.step()
    return loss.item()

run_training(config, train_step)
```

上述 `model`、`torch` 和 `compute_batch_loss` 需由具体训练实现提供。
骨架只设置 Python random 种子；接入 PyTorch/NumPy 时须在模型初始化前自行设置相应种子。
目前不包含 checkpoint、恢复训练或验证集评估。
