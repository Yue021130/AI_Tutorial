# DETR (DEtection TRansformer) 学习资源合集

本项目收集了 DETR（Detection Transformer）相关的学习资料和代码，包括官方 Notebook 示例和第三方训练框架。

## 简介

DETR 是由 **Meta AI (Facebook AI)** 于 2020 年提出的**端到端目标检测模型**。它首次将 Transformer 架构成功应用于目标检测任务，摒弃了传统检测器中复杂的手动设计组件（如 Anchor、NMS 等），实现了真正的端到端检测。

---

## 核心思想

传统目标检测器（如 Faster R-CNN、YOLO）依赖大量手工设计的组件：
- Anchor 生成与匹配
- 非极大值抑制 (NMS)
- 多尺度特征金字塔

**DETR 的核心创新**在于将目标检测视为一个**集合预测问题**（Set Prediction Problem）：
- 使用 CNN 提取图像特征
- 使用 Transformer Encoder-Decoder 建模全局关系
- 使用二分匹配（Bipartite Matching）计算损失
- 直接输出固定数量的预测框，无需后处理

---

## 模型架构

```
输入图像 → CNN Backbone (ResNet-50) → 特征图
                                    ↓
                         Transformer Encoder (自注意力)
                                    ↓
                         Transformer Decoder (交叉注意力)
                                    ↓
                         并行 Object Queries (100个可学习查询)
                                    ↓
                         共享 FFN → 输出类别 + 边界框坐标
```

### 关键组件

| 组件 | 说明 |
|------|------|
| **Backbone** | 通常使用 ResNet-50/101 提取图像特征 |
| **Transformer Encoder** | 全局自注意力，建模像素间关系 |
| **Transformer Decoder** | 接收 Object Queries，与编码器特征交叉注意力 |
| **Object Queries** | 100 个可学习的嵌入向量，每个代表一个候选目标 |
| **Prediction Heads** | 共享的前馈网络，输出类别概率和归一化边界框坐标 |

---

## 核心机制

### 1. 二分匹配损失 (Hungarian Loss)

DETR 输出固定数量 N 个预测（通常 N=100），使用 **匈牙利算法** 将预测与真实目标进行最优一对一匹配：

```
匹配成本 = λ_cls · 分类损失 + λ_L1 · 边界框L1损失 + λ_giou · GIoU损失
```

### 2. 并行解码

不同于 NLP 中 Transformer 的自回归解码，DETR 的 Object Queries **同时并行处理**，推理速度更快。

### 3. 全局关系建模

自注意力机制天然具备全局感受野，对大目标和遮挡场景更友好。

---

## 主要特点

| 优势 | 劣势 |
|------|------|
| 真正端到端，无需 NMS | 小目标检测性能较弱 |
| 架构简洁，易于扩展 | 训练收敛慢（需 500 epochs） |
| 全局上下文建模能力强 | 查询数量固定，计算量较大 |
| 可自然扩展至全景分割 | 对超参数敏感 |

---

## 项目文件说明

### 📓 官方 Notebook 示例

以下三个 Notebook 来源于 **Meta AI (Facebook) 官方 DETR 仓库** [facebookresearch/detr](https://github.com/facebookresearch/detr)：

| 文件 | 说明 | 来源 |
|------|------|------|
| `detr_demo.ipynb` | DETR 最小化实现演示，包含模型定义、预训练权重加载、预测可视化 | [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| `detr_hands_on.ipynb` | DETR 动手实践教程，展示如何使用预训练模型进行目标检测及注意力可视化 | [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| `DETR_panoptic.ipynb` | DETR 全景分割（Panoptic Segmentation）示例，使用 Detectron2 进行可视化 | [facebookresearch/detr](https://github.com/facebookresearch/detr) |

### 🏭 第三方训练框架

| 目录 | 说明 | 来源 |
|------|------|------|
| `DETR-Factory-PyTorch/` | DETR & Conditional DETR 训练/微调/评测工程框架，支持在自定义 COCO 格式数据集上训练 | 第三方开源项目（基于 DETR 论文实现） |

---

## 应用场景

- 通用目标检测
- 全景分割（Panoptic Segmentation）
- 作为基础模型扩展（如 Deformable DETR、DINO、DETR 3D 等）

---

## 参考资料

- 论文: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) (ECCV 2020)
- 官方仓库: [facebookresearch/detr](https://github.com/facebookresearch/detr)
- 后续改进: Deformable DETR, Conditional DETR, DAB-DETR, DINO 等

---

> **提示**: 本目录下的 `.ipynb` 文件包含可运行的 DETR 示例代码，建议使用 Jupyter Notebook 或 VS Code 打开查看。
