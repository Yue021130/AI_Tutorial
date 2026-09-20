# 02 · 论文精读：Masked Autoencoders Are Scalable Vision Learners

## 0. 论文信息

- **标题**：Masked Autoencoders Are Scalable Vision Learners
- **作者**：Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick（Facebook AI Research, FAIR）
- **出处**：arXiv:2111.06377（2021-11）；CVPR 2022（oral）
- **官方代码**：github.com/facebookresearch/mae
- **一句话**：把 BERT 式掩码预训练以「非对称 encoder-decoder + 75% 随机掩码 + 像素回归」的形式高效搬到视觉，仅用 ImageNet-1k 预训练，把 ViT-Huge 微调到 87.8% top-1（当时 IN-1k 上的最高水平之一）。

## 1. 核心主张（三件套）

1. **非对称 encoder-decoder**：encoder 只处理可见 patch（25%），decoder 处理全部 patch 但轻量且预训练后即丢弃。掩码率越高，encoder 计算量越省——**掩码第一次成为加速手段而非负担**。
2. **75% 随机掩码**：远高于 BERT 的 15%。图像冗余度高，只有高掩码率才能逼出整体推理；且随机掩码优于块状（block）掩码。
3. **像素回归目标**：对被掩 patch 预测逐 patch 归一化后的像素，MSE 损失，只算 masked 位置。不需要 tokenizer、不需要对比学习的配套工程。

## 2. 方法

### 2.1 掩码策略

- **均匀随机**选 75% 的 patch 置为被掩；每个 iteration 重新采样。
- 与 block / grid 掩码的对比：**随机掩码效果最好**（消融见 §4）。解释：随机掩码产生高度不完整、不连续的可见集合，局部外推失效，必须做全局语义推理；block 掩码留下大片完整区域，反而"可抄"。
- 与直觉"块状掩码更难所以更好"相反——这是论文里最反共识的发现之一。

### 2.2 Encoder：只看可见 patch

- 标准 ViT。输入 224×224、patch 16 → $N=196$，掩 75% 后 encoder 只处理 **49 个 token**。
- **不引入 mask token**：被掩 patch 根本不进入 encoder。序列变短 → 计算与显存大幅下降；attention 项按 $N^2$ 计算的收益更大。
- 副产品：**预训练与微调无分布错配**。BERT 的 encoder 预训练时见过 `[MASK]` token，微调时没有；MAE 的 encoder 两个阶段看到的输入形态一致。
- ViT-H/14 的设定：$N=256$，可见 64 token。

### 2.3 Decoder：轻量、全 token、用完即弃

- 输入 = 可见 patch 的 encoder 输出（经线性投影到解码器维度）+ **共享可学习的 mask token**，按原图顺序排好，各自加位置编码。
- **mask token 必须加位置编码**，否则模型不知道每个空位对应图像哪个位置。
- 论文 ImageNet 实验的默认 decoder：**宽度 512、深度 8 层**——相对 ViT-L encoder（1024 宽、24 深）是零头。decoder 只在预训练存在，下游用不到，可以放心做小。
- 消融显示 decoder 容量的作用高度依赖评估方式（见 §4.2）——这是论文最有洞察力的分析之一。

### 2.4 重构目标

- 每个 masked patch 回归 $3 p^2 = 768$ 维像素向量。
- **逐 patch 归一化**（减均值、除标准差）后再算 MSE，略优于原始像素——归一化抹掉了 patch 间的光照/对比度差异，让损失更关注结构。
- 损失**只对 masked patch 计算**（与 BERT 一致）；visible patch 已在 encoder 输入里，重建它们没有监督价值。
- 与 BEiT 离散 token 目标的比较见 §4.4。

## 3. 训练配置（论文/官方仓库口径）

| 项 | 值 | 备注 |
| --- | --- | --- |
| 数据 | ImageNet-1k 训练集（无标签使用） | 不用 IN-21k/JFT |
| 输入 / patch | 224 / 16（ViT-H 用 14） | |
| mask ratio | 0.75 | |
| decoder | dim 512, depth 8 | 默认 |
| 优化器 | AdamW | betas（ViT-L/H 用 (0.9, 0.95)，待核验） |
| lr | base 1.5e-4，按 batch/256 线性缩放 | |
| weight decay | 0.05 | |
| batch size | 4096（主实验） | 小 batch 需同步缩 lr |
| epochs | 主结果 1600（ViT-L/H） | 消融用更短 schedule（具体待核验） |
| warmup / 衰减 | warmup 数十 epoch + cosine（具体待核验） | |
| 增广 | RandomResizedCrop + 随机翻转 | **无 color jitter**（见 §4.5） |
| 精度 | fp16 / bf16 混合精度（待核验） | |

> 上表中凡标「待核验」处以官方仓库脚本为准；数值量级（1.5e-4 / 0.05 / 4096 / 1600）是可靠的。

## 4. 消融实验（论文的核心证据）

### 4.1 mask ratio

- **fine-tune**：40%–80% 之间平缓，75% 附近最好；90% 时明显退化（任务超出可恢复范围）。
- **linear probe**：对掩码率敏感得多，**峰值约在 75%**，比 fine-tune 的曲线尖锐。
- 两类评估的分歧是贯穿全文的主题：linear probe 度量"表征本身线性可分性"，fine-tune 度量"表征作为初始化的潜力"。MAE 的主张是后者才是重点。

### 4.2 decoder 深度与宽度

- **fine-tune 场景：decoder 1 层即接近最优**——下游只保留 encoder，decoder 只需把 encoder 输出"翻译"成像素即可。
- **linear probe 场景：需要约 8 层**。解释：linear probe 只能拿 encoder 最后一层输出，decoder 越深，预训练时"抽象"发生得越靠近 encoder 顶端，表征越线性可分。
- 宽度：256 已基本够用，512 是稳妥默认。

### 4.3 mask token 的必要性

- 消融去掉 mask token（用 encoder 输出填充空位等替代方案）后：fine-tune 几乎不受影响，linear probe 明显下降（方向待核验）。说明 mask token 主要服务于"补全任务"本身，对 encoder 表征质量不是必需——非对称结构的极限可以更进一步（SimMIM/CAE 沿此方向，见 06）。

### 4.4 重构目标的选择

- 候选：原始像素 / 归一化像素 / PCA 主成分空间 / BEiT 式离散 token。
- 结论：**归一化像素综合最好**；离散 token 目标在 linear probe 上更好（语义更强），但 fine-tune 不占优，且需要两阶段 tokenizer（方向性结论待核验具体排序）。
- 论文立场：既然目标是通用预训练（以 fine-tune 论英雄），像素目标的简单性胜出。

### 4.5 数据增强

- **只用 crop + flip 即可**；加 color jitter 无益甚至略降。
- 解释：**随机掩码本身就是极强的增广**——每个 epoch 每张图的掩码都不同，有效任务空间天文数字级，外置增广的边际价值被摊薄。这也是 MAE 训练管线极简的原因。

### 4.6 掩码策略

- random > grid > block（block 明显更差），见 §2.1。

## 5. 主结果与迁移实验

### 5.1 ImageNet-1k 微调

| 模型 | top-1（fine-tune, 224/512） | 备注 |
| --- | --- | --- |
| ViT-B/16 | 83.6 | |
| ViT-L/16 | ≈85.9（待核验） | |
| ViT-H/14 | **87.8** | 仅 IN-1k 预训练 + 微调 |

- linear probe 明显低于对比方法（ViT-L 约 65 上下，待核验）——论文并不回避这一点，而是用它论证 linear probe 不是好的表征度量。
- **scaling 无饱和迹象**：ViT-B → L → H 单调提升；模型越大，MAE 相对监督训练的优势越大。

### 5.2 下游任务（MAE 真正的主场）

- **COCO 检测/实例分割**（ViT-L + Mask R-CNN）：相对监督预训练基线提升约 3–4 个 AP（具体数值待核验）。
- **ADE20K 语义分割**（ViT-L + UperNet）：mIoU 提升约 3 个点以上（待核验）。
- 一致规律：**任务越"密集"（逐空间位置预测），MAE 增益越大**——重构式预训练天然对齐空间结构理解，而对比学习的实例级目标与之错位。

### 5.3 训练效率

- encoder 只算 25% token + decoder 轻量 → 论文报告相对 BEiT 等有数倍墙钟加速（约 3–3.7×，具体倍数待核验）。
- 代价是超长 epoch（1600）：总训练时长仍可观，但**每单位性能的计算成本显著更低**。

## 6. 评述

**优点**
- 简单：单阶段、无 tokenizer、无负样本、无 EMA、无花式增广。
- 快：非对称结构把掩码变成加速器。
- 可扩展：模型、数据双双可放大且不饱和。
- 下游强：密集任务增益显著。

**局限**
- linear probe 弱：特征线性可分性差，**必须整体微调**才能兑现性能（对算力不足的用户不友好）。
- 目标偏底层：像素重构的监督信号天花板可能低于语义级目标（CAE/MaskFeat 的改进方向）。
- 高掩码率在低分辨率/小数据上未必最优；预训练 epoch 数巨大。

## 7. 精读自测题

1. 为什么 decoder 对 fine-tune 可以只有 1 层，而 linear probe 需要 8 层？
2. 75% vs 15% 的差异根源是什么？为什么 linear probe 的最优掩码率曲线比 fine-tune 尖锐？
3. 非对称结构到底省了多少计算？encoder 49 token vs 196 token，FLOPs 各项怎么变？
4. mask token 对谁重要、对谁不重要？这个消融预示了哪些后续工作？
5. 为什么 color jitter 对 MAE 有害，而对对比学习几乎必需？

带着这些问题读 `03-formulation.md`（数学）与 `04-implementation.md`（代码）。
