# 上游来源说明

本项目在原有视觉语言跟踪框架上实现并整理 RSPTrack。项目命名调整不改变代码来源，保留各文件已有的版权和许可证头部。

- SAVLT：Self-Adaptive Vision-Language Tracking with Context Prompting，原项目框架和文本编码接口来源；[论文](https://ieeexplore.ieee.org/document/11284903)。
- [SUTrack](https://github.com/chenxin-dlut/SUTrack)：训练、跟踪和评估框架的相关实现。
- [CoCoOP / CoOp](https://github.com/KaiyangZhou/CoOp)：上游的文本提示相关实现；当前论文配置关闭 CoOp 提示。
- [iTPN / Fast-iTPN](https://github.com/sunsmarterjie/iTPN)：HiViT / Fast-iTPN 编码器实现，保留原 MIT 声明。
- [OpenAI CLIP](https://github.com/openai/CLIP)：ViT-L/14 文本编码器。
- Quest / ReSA：稀疏注意力模块原实现说明中的块描述与查询相关检索来源。RSPTrack 的 LRSA 面向双向联合编码，在层间周期性执行密集交互，未使用自回归 KV 缓存刷新。
- 其余代码沿用文件中的来源声明，包括 DINOv2、timm 及跟踪数据加载 / 评估组件。

当前提供的代码目录未含完整的上游 LICENSE 集合。本次整理未擅自为整个派生项目指定新许可证；公开发布时应补齐实际使用版本的上游许可文件，并保留其要求的声明。
