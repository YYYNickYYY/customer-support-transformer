# VS Code 中使用 GitHub Pull Request

## 一次性设置

1. 在 VS Code 打开本项目，安装推荐扩展 GitHub Pull Requests。
2. 点击账户图标登录 GitHub，按照浏览器提示完成授权。密码/token 不写入项目文件。
3. 若尚未上传：源代码管理中的 Publish to GitHub 可创建仓库；先确认公开/私有设置。
4. 若使用现有仓库：添加它为 origin，然后发布 main。已有内容的远程仓库需先拉取并检查，不强制覆盖。

## 每次修改的流程

1. 切换到 main，执行 Pull 拉取最新内容。
2. 点击左下角分支名 → Create new branch，命名如 `feat/tiny-transformer`。
3. 修改代码并运行 `.\.venv\Scripts\python.exe -m pytest -q`。
4. 在源代码管理检查差异，Stage Changes，填写说明并 Commit。
5. 点击 Publish Branch/Push 上传分支。
6. 命令面板运行 `GitHub Pull Requests: Create Pull Request`，目标分支选择 main。
7. 填写 PR 模板；查看 GitHub Actions 测试结果，再进行 Review/Merge。
8. 合并后切回 main 并 Pull。

冻结的 data/splits 和 artifacts 文件通过 .gitattributes 保留原始字节，避免换行转换影响哈希。
远程 PR 检查需要 GitHub Actions 可用；要强制检查通过才能合并，可在仓库设置中为 main 配置分支保护。
