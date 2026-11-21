# RSPTrack：GitHub 代码与 checkpoint 发布流程

## 1. 发布方式

建议代码放普通 Git 仓库，训练所得推理权重放同一仓库的 **Releases 附件**。读者可先克隆代码，再下载权重，避免每次克隆都拉取大文件。

GitHub 普通 Git 会拒绝超过 100 MiB 的单文件；网页直接添加代码文件的上限为 25 MiB。Release 的每个附件必须小于 2 GiB。[大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Release 说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。这些限制按 2026-11-19 查询的文档记录。

本目录已加入 `.gitignore`，忽略权重、数据、输出、私人路径配置和缓存。请在 `Code-上传` 内初始化 Git，仓库根目录应直接看到 README、lib、tracking 和 experiments。

## 2. 创建空仓库并推送代码

先安装 Git，登录 GitHub，在网页右上角 New repository 创建名为 `RSPTrack` 的空仓库。需要供论文读者访问时选择 Public；创建时不要勾选自动生成 README、LICENSE 或 .gitignore，以免与本地已有文件冲突。

Windows PowerShell 中执行，替换用户名、邮箱及仓库地址：

```powershell
Set-Location -LiteralPath 'D:\论文撰写-2\Code-上传'
git init -b main
git config user.name "你的 GitHub 用户名"
git config user.email "你的 GitHub 提交邮箱"
git add .
git status --short
git diff --cached --stat
git commit -m "Release RSPTrack code and paper configurations"
git remote add origin https://github.com/YOUR_USERNAME/RSPTrack.git
git push -u origin main
```

执行前确认目录确为 `Code-上传`。本次整理未替你创建仓库、提交或联网推送。认证时使用 Git Credential Manager 的浏览器登录，或 GitHub 允许的令牌 / SSH 方式。[GitHub 官方导入流程](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows)。

## 3. 准备权重附件

在原训练环境执行：

```bash
python tracking/transfer_model.py --input /path/to/SAVLT_ep0200.pth.tar --output release/RSPTrack_ep0200.pth.tar --config RSPTrack-B
```

旧权重文件名可作为输入；必须确认它来自完整 TVPM + LRSA、α=0.5、f=4 的训练配置。该工具不会把其他消融结果改造成完整模型。默认去除优化器等训练状态，不改变张量精度，生成：

```text
release/RSPTrack_ep0200.pth.tar
release/RSPTrack_ep0200.pth.tar.yaml
release/RSPTrack_ep0200.pth.tar.sha256
```

还应保存真正用于实验的环境清单：

```bash
pip freeze > release/requirements-training.txt
```

在发布说明中写明 GPU、驱动、CUDA、训练轮数、配置名、文本协议及逐项评估结果。发布多个消融权重时，分别导出到独立目录，上传前给附件加配置前缀以避免同名；读者下载后可使用 `--checkpoint` 指定。

如果需要公开可续训 checkpoint，可加 `--keep-training-state` 另行导出；它更大，也会包含训练设置对象。常规论文复现使用精简推理权重即可。

## 4. 创建 Release 并上传

在 GitHub 仓库页面进入 Releases → Draft a new release：

1. 新建标签 `v1.0.0`，目标选择已推送的 `main`。
2. 标题填写 `RSPTrack v1.0.0`。
3. 附件区拖入推理权重、配置、校验值和环境清单。
4. 发布说明写明权重对应完整模型，LaSOT / LaSOT-ext 使用占位文本，TNL2K 使用真实描述；OTB99-lang 的论文补充结果对应单独的 LRSA 变体。
5. 发布后复制附件的真实下载地址，填写到 README 的权重部分，并标注 SHA-256。

如已安装 GitHub CLI，也可以在代码推送后创建草稿：

```bash
gh auth login
gh release create v1.0.0 --target main --title "RSPTrack v1.0.0" --draft
gh release upload v1.0.0 release/RSPTrack_ep0200.pth.tar release/RSPTrack_ep0200.pth.tar.yaml release/RSPTrack_ep0200.pth.tar.sha256 release/requirements-training.txt
```

随后在网页核对说明并发布。不要把 draft Release 的地址当成公开下载地址。

每个附件必须小于 2 GiB。如果精简权重仍超限，可使用 Git LFS 或可信模型托管服务，并在 Release 给出链接。不要仅改扩展名继续普通 Git 推送。Git LFS 会受套餐的单文件限制和存储 / 下载配额约束，采用前查看 [Git LFS 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

## 5. 验证读者使用流程

在干净目录克隆新仓库，按 README 安装环境和配置路径；从公开 Release 下载权重，放到：

```text
output/checkpoints/train/rsptrack/RSPTrack-B/RSPTrack_ep0200.pth.tar
```

先运行一个序列，再运行完整基准：

```bash
python tracking/test.py rsptrack RSPTrack-B --dataset_name lasot_extension_subset --sequence 序列名
python tracking/test.py rsptrack RSPTrack-B
python tracking/analysis_results.py
```

Windows 可用以下命令核对下载文件校验值：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'output/checkpoints/train/rsptrack/RSPTrack-B/RSPTrack_ep0200.pth.tar'
```

与 `.sha256` 文本中的值一致后，确认整个数据集评估的结果与论文对应。只重命名、编译通过或单序列跑通都不能替代这一验证。

## 6. 论文中的链接

正文代码链接使用仓库主页，例如 `https://github.com/YOUR_USERNAME/RSPTrack`。在 README 记录用于论文的 tag 和 commit；权重链接指向对应的 Release。LaTeX 示例：

```latex
代码与训练权重已公开于 \url{https://github.com/YOUR_USERNAME/RSPTrack}。
```

仓库实际公开且附件可下载后，再替换论文 `[URL]`。目前没有你的 GitHub 用户名和仓库地址，本次未擅自填写链接或宣称已经公开。

后续更新流程：修改文件 → `git add .` → `git commit -m "说明本次修改"` → `git push`。论文定稿对应的权重版本应保留；需要更换权重时发布新 tag，并说明差异。

## 7. Git LFS 备选方案

如果你明确希望权重随仓库下载，可安装 Git LFS，并在首次添加权重前配置：

```bash
git lfs install
git lfs track "published_checkpoints/*.pth.tar"
git add .gitattributes
git add -f published_checkpoints/RSPTrack_ep0200.pth.tar
git commit -m "Add RSPTrack checkpoint with Git LFS"
git push
```

这里的 `-f` 仅用于显式选定的权重文件，因为项目 `.gitignore` 默认忽略所有 checkpoint。不要强制添加整个 output 目录。Release 附件方案通常更便于代码克隆与版本管理。
