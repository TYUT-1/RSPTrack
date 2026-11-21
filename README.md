# RSPTrack：语言辅助的校正稀疏视觉跟踪

RSPTrack 在融合前使用 **TVPM（Target-aware Visual Prompt Modulation，目标感知视觉提示调制）**，在联合编码中使用 **LRSA（Layer-wise Rectified Sparse Attention，层间校正稀疏注意力）**。本仓库对应论文中文稿的完整模型和消融配置。

## 方法与实现

- 视觉编码器：B 型 HiViT，代码工厂名 `fastitpnb`，通道数 512、步长 16。
- 文本编码器：CLIP ViT-L/14，冻结 CLIP 主体，保留可训练投影层。
- 模板 / 搜索输入：112 × 112 / 224 × 224；联合序列为 49 + 196 + 1 个类别 token + 1 个文本 token = 247。
- TVPM：隐藏维度 512，角色差分系数 α = 0.5；通过残差调制特征，不追加提示 token。
- LRSA：24 个融合主块、8 个注意力头；键块长度 16，初始保留比例 0.5，最少 4 块，局部保留 1 块，密集校正间隔 f = 4（18 个稀疏层 + 6 个密集层）。局部和受保护块可使实际保留比例高于 0.5。
- 中心式定位头，GIoU / L1 / focal 跟踪损失；推理使用 Hann 窗，无在线模板更新。

LRSA 参考实现采用密集注意力矩阵加掩码，保留完整 token 序列。它用于验证信息交互方式，不能直接据此宣称稀疏算子的速度或显存收益。

## 安装

训练和跟踪需要 NVIDIA GPU；多卡训练使用 Linux / NCCL。以下是一组安装起点，**不是原实验机环境的完整锁定记录**。发布复现版本时请附上实际训练环境的 `pip freeze`、GPU 和 CUDA 信息。

```bash
conda create -n rsptrack python=3.10 -y
conda activate rsptrack
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

PyTorch 与 CUDA 的其他匹配组合见 [PyTorch 官方安装记录](https://pytorch.org/get-started/previous-versions/)。`jpeg4py` 需要系统提供 libjpeg-turbo。默认训练使用 OpenCV 图像加载，但导入数据模块时仍需要 jpeg4py 包。

## 配置数据路径

所有命令在仓库根目录执行。首次配置：

```bash
python tracking/create_default_local_file.py --data_dir /path/to/SOT_datasets --save_dir /path/to/RSPTrack/output
```

这会生成被 Git 忽略的 `paths.local.json`。可参照 `paths.example.json` 为已有目录逐项设置路径；无需修改 Python 文件。也可使用 `RSPTRACK_DATA_DIR`、`RSPTRACK_SAVE_DIR` 或 `RSPTRACK_PATHS_FILE` 环境变量。未配置时默认使用仓库内 `data/` 和 `output/`。

默认目录约定：

```text
SOT_datasets/
├── lasot_lmdb/
├── got10k_lmdb/
├── coco_lmdb/
├── trackingnet_lmdb/
├── TNL2K/
│   ├── train_set/<sequence>/imgs/、groundtruth.txt、language.txt
│   └── test_set/<sequence>/imgs/、groundtruth.txt、language.txt
├── lasot/<class>/<sequence>/img/、groundtruth.txt、full_occlusion.txt、out_of_view.txt
├── LaSOT_Ext/<class>/<sequence>/img/、groundtruth.txt、full_occlusion.txt、out_of_view.txt
└── otb_lang/OTB_videos/、OTB_query_test/
```

`--use_lmdb 1` 对 LaSOT、GOT-10k、COCO 和 TrackingNet 使用 LMDB，TNL2K 仍从原始文件读取。LMDB 的相对键布局必须与数据加载器一致。训练划分来自 `lib/train/data_specs/`，其中 GOT-10k 使用 `GOT10K_vottrain` 划分。本模型采用五数据集混合训练，不能作为 GOT-10k 官方仅训练集协议的结果。

LaSOT LMDB 加载器未实现真实描述读取，因此所有论文配置均显式设置 `DATA.USE_NLP.LASOT: false`。若误改为 `true`，加载器会直接报错，防止配置宣称使用真实文本却实际静默使用占位输入。需要真实描述训练时，应先实现并验证对应的文本加载流程。

## 预训练参数与训练权重

训练前放置视觉初始化权重：

```text
pretrained/itpn/fast_itpn_base_clipl_e1600.pt
```

CLIP ViT-L/14 由 OpenAI CLIP 的 `clip.load` 下载并缓存；离线运行需提前准备缓存。视觉初始化文件的来源与校验值应随正式 Release 一并记录，当前目录未附带该文件。已有完整训练权重时，测试不会再次读取视觉初始化文件，但仍会构建 CLIP 文本编码器。

正式训练权重请从本仓库 **Releases** 下载；当前整理版本尚未附带权重，也未填写不存在的下载链接。默认完整模型的放置位置为：

```text
output/checkpoints/train/rsptrack/RSPTrack-B/RSPTrack_ep0200.pth.tar
```

也可以测试时通过 `--checkpoint /path/to/file.pth.tar` 指定文件。请使用自己训练或可信来源的 checkpoint。

## 训练

```bash
python tracking/train.py
```

默认 `rsptrack / RSPTrack-B`，同时启用 TVPM 和 LRSA，启用 LMDB。五类数据 LaSOT、TNL2K、GOT-10k、COCO2017、TrackingNet 等比例采样，每轮构建 60,000 个训练对。**TNL2K 使用真实文本，其余四类使用长度 77 的全零 token ID 序列；占位文本仍经文本编码器产生一个融合 token。**

优化器为 AdamW，学习率与权重衰减均为 1e-4，编码器学习率倍率 0.1；单卡 batch size 32；200 轮，第 150 轮后学习率乘 0.1。模板 / 搜索裁剪因子分别为 2.0 / 4.0。每 10 轮及最终轮保存 checkpoint，同一配置下自动恢复最新 checkpoint；新实验请使用不同配置或保存目录。

多卡示例：

```bash
python tracking/train.py --mode multiple --nproc_per_node 2
```

配置中的 batch size 是每个进程的值；两卡默认总 batch size 为 64。若要维持论文总 batch size 32，应复制实验配置并将每卡值改为 16，避免直接覆盖原配置。

## 测试与评估

完整模型默认测试 LaSOT-ext，随后评估同一配置的预测：

```bash
python tracking/test.py rsptrack RSPTrack-B
python tracking/analysis_results.py
```

你原先使用的命令仍可运行，并会映射到上述新名称与新结果目录：

```bash
python tracking/test.py savlt SAVLT-B-ReSA-TVPM_512_05
```

历史名称只用于兼容参数及来源说明；模型实现、配置、默认输出和展示名称均使用 RSPTrack。旧命令不会自动搜索旧目录中的权重或预测结果，旧 checkpoint 可使用 `--checkpoint` 指定，或按下文导出到新目录。

| 基准 | 数据集参数 | 文本条件 | 模型配置 |
| --- | --- | --- | --- |
| LaSOT | `lasot` | 固定占位文本 | `RSPTrack-B` |
| LaSOT-ext | `lasot_extension_subset` | 固定占位文本 | `RSPTrack-B` |
| TNL2K | `tnl2k` | 数据集真实描述 | `RSPTrack-B` |
| OTB99-lang（补充） | `otb99_lang` | 数据集真实描述 | `RSPTrack-B-LRSA`（关闭 TVPM） |

LaSOT 与 LaSOT-ext 不需要真实文本目录或 `language.txt`。`lasot_lang`、`lasot_ext_lang` 及其语言初始化分支已移除，直接使用表中的视觉基准入口。

```bash
python tracking/test.py rsptrack RSPTrack-B --dataset_name lasot
python tracking/analysis_results.py --dataset_name lasot
python tracking/test.py rsptrack RSPTrack-B --dataset_name tnl2k
python tracking/analysis_results.py --dataset_name tnl2k
python tracking/test.py rsptrack RSPTrack-B-LRSA --dataset_name otb99_lang
python tracking/analysis_results.py --config RSPTrack-B-LRSA --dataset_name otb99_lang
```

可以添加 `--sequence <序列名>` 单序列调试，或设置 `--threads` 与 `--num_gpus`。评估完整基准前必须先生成所有序列结果，缺失结果会报错。结果保存于 `output/test/tracking_results/rsptrack/<配置>/<数据集>/`；评估缓存位于 `output/test/result_plots/`。训练时如果另设 `--save_dir`，请同步修改路径配置的 `save_dir`，使测试与评估读取同一目录。`--plot` 额外导出 PDF 曲线。

评估输出包括成功率 AUC、20 像素精度 P、归一化距离阈值 0.2 下的 P_norm。

## 论文结果

以下数值来自所给论文中文稿，单位为百分比，**不代表当前机器重新运行验证的结果**。

| 模型 | 基准 | AUC | P_norm | P |
| --- | --- | ---: | ---: | ---: |
| RSPTrack | LaSOT | 72.2 | 82.0 | 78.9 |
| RSPTrack | LaSOT-ext | 52.8 | 63.5 | 60.5 |
| RSPTrack | TNL2K | 65.2 | 82.7 | 70.1 |
| RSPTrack（关闭 TVPM） | OTB99-lang | 69.5 | 83.3 | 91.3 |

## 消融配置

所有配置位于 `experiments/rsptrack/`。训练时用 `--config <配置>`，测试第二个位置参数及评估 `--config` 必须使用同一配置名，并放置该配置对应的权重。

| 配置 | TVPM | LRSA | 说明 |
| --- | --- | --- | --- |
| `RSPTrack-B-Baseline` | 关 | 关 | 密集基线 |
| `RSPTrack-B-TVPM` | 开 | 关 | Baseline + TVPM |
| `RSPTrack-B-LRSA` | 关 | 开 | Baseline + LRSA；OTB99-lang 补充实验 |
| `RSPTrack-B` | 开 | 开 | 完整模型，α=0.5、f=4 |
| `RSPTrack-B-TVPM-alpha0` | 开 | 开 | α=0 |
| `RSPTrack-B-TVPM-alpha025` | 开 | 开 | α=0.25 |
| `RSPTrack-B-TVPM-alpha1` | 开 | 开 | α=1 |
| `RSPTrack-B-LRSA-f0` | 开 | 开 | 0 个密集层、24 个稀疏层 |
| `RSPTrack-B-LRSA-f2` | 开 | 开 | 12 个密集层、12 个稀疏层；论文结果待补 |
| `RSPTrack-B-LRSA-f8` | 开 | 开 | 3 个密集层、21 个稀疏层；论文结果待补 |

## 导出与发布 checkpoint

从自己的完整训练 checkpoint 导出推理文件（输出文件必须尚不存在）：

```bash
python tracking/transfer_model.py --input /path/to/your/checkpoint.pth.tar --output release/RSPTrack_ep0200.pth.tar --config RSPTrack-B
```

导出文件保留张量精度与参数键，剥离优化器、训练设置对象和统计记录，并生成 `.sha256` 校验文件及 `.yaml` 配置副本。旧类名和文件名不会影响张量加载；重命名本身不会把其他消融的权重变为完整模型。请确保输入确为该配置的第 200 轮权重，核对导出的 epoch 后再使用默认文件名。

若需要把旧权重迁移到新路径继续训练，额外使用 `--keep-training-state`，将输出放到对应 `output/checkpoints/train/rsptrack/<配置>/RSPTrack_epXXXX.pth.tar`。它会保留优化器和统计信息并更新模型类型；此类文件适合续训，正式推理发布使用默认精简导出。

完整 GitHub 建库、推送、Release 上传和论文链接填写流程见 [GitHub 发布指南](docs/GITHUB发布指南.md)。权重默认被 `.gitignore` 排除，不会被普通 `git add .` 加入代码历史。

## 本地检查

安装依赖后可运行 CPU 检查：

```bash
python -m unittest discover -s tests -v
```

检查覆盖配置合并、消融开关、旧命令映射、LaSOT 无文本加载、训练 / 测试占位输入一致性、LRSA 掩码与梯度、权重导出以及合成序列的评估 / PDF 导出。它们不能替代真实数据上的完整训练和精度复现。

## 来源与引用

代码框架基于 SAVLT，并沿用 SUTrack、CoCoOP、iTPN/HiViT、CLIP 等工作的相关实现；保留来源声明和上游文件中的版权说明。LRSA 的查询相关块检索参考 Quest/ReSA 思路，具体区别见代码说明。感谢这些工作的公开实现。详细来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

RSPTrack 的正式英文题名、作者、发表信息及 BibTeX 在论文定稿后补充。当前未提供这些元数据，故不生成虚构引用，也不将 SAVLT 论文冒充本项目论文。
