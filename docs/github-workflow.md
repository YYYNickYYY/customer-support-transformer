# VS Code 中使用 GitHub Pull Request

## 一次性设置

1. 在 VS Code 打开本项目，安装推荐扩展 GitHub Pull Requests。
2. 点击账户图标登录 GitHub，按照浏览器提示完成授权。密码/token 不写入项目文件。
3. 若尚未上传：源代码管理中的 Publish to GitHub 可创建仓库；先确认公开/私有设置。
4. 若使用现有仓库：添加它为 origin，然后发布 main。已有内容的远程仓库需先拉取并检查，不强制覆盖。

## 每次修改的流程

1. 切换到 main，执行 Pull 拉取最新内容。
2. 点击左下角分支名 → Create new branch，命名如 `codex/tiny-transformer`。
3. 修改代码并运行 `.\.venv\Scripts\python.exe -m pytest -q`。
4. 在源代码管理检查差异，Stage Changes，填写说明并 Commit。
5. 点击 Publish Branch/Push 上传分支。
6. 命令面板运行 `GitHub Pull Requests: Create Pull Request`，目标分支选择 main。
7. 填写 PR 模板；查看 GitHub Actions 测试结果，再进行 Review/Merge。
8. 合并后切回 main 并 Pull。

冻结的 data/splits 和 artifacts 文件通过 .gitattributes 保留原始字节，避免换行转换影响哈希。
远程 PR 检查需要 GitHub Actions 可用；要强制检查通过才能合并，可在仓库设置中为 main 配置分支保护。

## 本次发布与本地产物边界

仓库：`https://github.com/YYYNickYYY/tiny-transformer-lab`，PR目标为`main`。仓库从my-project-1更名，Python包名和历史路径保持不变。PR创建后先检查Tests，不自动合并或强制覆盖main。

- 按用户要求，`*.code-workspace`不上传，本地工作台保留。`.vscode/`是共享编辑器配置和历史任务，仍随源码发布。
- `runs/`、`outputs/`、`.venv/`、`.tools/`及缓存不上传；不要把token、私人配置或环境文件加入提交。
- 新电脑运行`uv sync --locked --extra dev --extra data`后执行`python -m pytest -q`。没有整个本地outputs目录时，仅明确列出的历史回放测试跳过；常规模型/数据/生成测试仍执行。
- `tests/conftest.py`列出新增的本地产物依赖边界，不改旧冻结测试字节。现有但不完整的outputs目录不按“源码检出”跳过；错误哈希和损坏文件仍会失败。原历史测试自身已有的skip规则保留。
- 完整本地实验验收与源码检出测试应分别报告通过/跳过数量，不能把CI通过称为重跑历史训练或独立评估。
- `.gitattributes`保留冻结代码、配置、文档和数据的原始换行。发布前执行`python scripts/verify_artifacts.py --staged`检查暂存区冻结文件，避免上传时改变哈希。

历史生成入口需要本地checkpoint，训练门控需要对应冻结证据；缺少这些文件时应报告缺失，不要删门控或自动启动训练。本轮优化已收尾。
