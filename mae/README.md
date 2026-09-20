# MAE 学习笔记：Masked Autoencoders

系统学习掩码自编码器（Masked Autoencoders, MAE）的资料集，围绕论文 *"Masked Autoencoders Are Scalable Vision Learners"*（He et al., arXiv:2111.06377, CVPR 2022）展开：从动机到论文精读，再到数学形式化、最小 PyTorch 实现、实验调参与相关工作对比。

## 目标

读完本目录后应当能够：

1. 说清 MAE 解决的两个核心问题：视觉信号的**信息冗余**（为什么敢掩 75%）与自监督预训练的**可扩展性**（为什么要非对称结构）。
2. 用数学语言完整写出 MAE 的 patch 化、随机掩码、非对称 encoder-decoder 与重构损失。
3. 读懂一份约 300 行的 PyTorch 最小实现，追踪每个张量的形状流转。
4. 知道关键超参（mask ratio、decoder 容量、lr/wd/epoch）怎么调、常见失败模式长什么样。
5. 把 MAE 放进 MIM（masked image modeling）家族（BEiT / SimMIM / CAE / MaskFeat / VideoMAE）的坐标系里比较。

## 前置

- **PyTorch**：`nn.Module`、autograd、`AdamW`、`Conv2d`。
- **ViT**：patch embedding、Transformer block（pre-norm）、位置编码（可学习 / sin-cos）。
- **自监督概览**：contrastive（SimCLR / MoCo）与 generative / masked modeling 两条路线的差异。
- **基础数学**：均匀采样、均值/方差归一化、MSE。

## 建议路线

```
01-motivation.md                                  为什么需要 MAE（信息冗余 + 可扩展性）
   ↓
02-paper.md                                       论文精读（方法 / 消融 / 迁移）
   ↓
03-formulation.md                                 数学形式化（对照论文记号）
   ↓
04-implementation.md  +  code/mae_minimal.py      代码级理解 + 冒烟测试
   ↓
05-experiments.md                                 调参、算力、失败模式
   ↓
06-related.md                                     MIM 家族对比
   ↓
07-open-questions.md                              开放问题与待验证假设
```

## 文件索引

| 文件 | 内容 |
| --- | --- |
| `01-motivation.md` | AE → DAE → MLM → MAE 的思想脉络；视觉冗余性与可扩展性两条主线 |
| `02-paper.md` | MAE 论文精读：非对称结构、75% 掩码、消融与迁移实验（不确定处标注「待核验」） |
| `03-formulation.md` | 形式化：patch embedding、掩码集合、非对称编解码器、重构损失（LaTeX 公式） |
| `04-implementation.md` | 最小实现的模块说明、张量形状追踪、与官方实现的差异 |
| `code/mae_minimal.py` | 可运行的 PyTorch 最小实现（关键行中文注释，随机数据冒烟测试） |
| `05-experiments.md` | 超参与调参维度、算力估计、常见失败模式、评估协议 |
| `06-related.md` | 与 BEiT、SimMIM、CAE、MaskFeat、VideoMAE 的对比 |
| `07-open-questions.md` | 开放问题与可验证假设（附实验设计） |

## 约定

- 中文为主，术语保留英文（masking ratio、linear probe、fine-tune 等）。
- 公式用 LaTeX 表示。
- 论文细节凡不确定处标注「待核验」，以原文（arXiv:2111.06377 及官方仓库 facebookresearch/mae）为准。
