# 03 · 形式化：MAE 的数学描述

本章把 MAE 写成严格的数学对象。记号尽量与论文一致，公式用 LaTeX。

## 0. 记号表

| 符号 | 含义 |
| --- | --- |
| $x \in \mathbb{R}^{3 \times H \times W}$ | 输入图像 |
| $p$ | patch 边长（论文 16，ViT-H 用 14） |
| $N = HW / p^2$ | patch 总数（$224/16$ 时 $N=196$；记 $g = H/p = W/p$ 为网格边长） |
| $x_i \in \mathbb{R}^{3p^2}$ | 第 $i$ 个 patch 展平后的像素向量，按行优先顺序编号 $i = 0,\dots,N-1$ |
| $\rho \in [0,1)$ | 掩码率（masking ratio，论文 $\rho = 0.75$） |
| $M \subset \{0,\dots,N-1\}$ | 被掩 patch 的下标集合，$\lvert M \rvert = \lfloor \rho N \rfloor$ |
| $V = \{0,\dots,N-1\} \setminus M$ | 可见 patch 集合，$\lvert V \rvert = K = N - \lvert M \rvert$ |
| $d$ / $d'$ | encoder / decoder 的表示维度 |
| $u_i, v_i$ | encoder / decoder 的位置编码 |

## 1. Patch 化与 Patch Embedding

把图像按 $g \times g$ 网格切成互不重叠的 patch 并行优先展开：

$$
x = \mathrm{patchify}(x) = (x_1, \dots, x_N), \qquad x_i \in \mathbb{R}^{3p^2}.
$$

每个 patch 经同一个线性投影（等价于 kernel = stride = $p$ 的卷积）得到 token，再加位置编码：

$$
z_i = W_{\mathrm{emb}}\, x_i + b_{\mathrm{emb}}, \qquad W_{\mathrm{emb}} \in \mathbb{R}^{d \times 3p^2},
$$
$$
\tilde z_i = z_i + u_i, \qquad i = 0, \dots, N-1 .
$$

MAE 的位置编码 $u_i$ 取**固定的二维 sin-cos 编码**（不可学习）：把 $d$ 维对半分给纵、横两个坐标轴，每个轴再用标准频率 $10000^{-2k/d}$ 的正余弦展开。要点：**patch 的空间身份完全由 $u_i$ 携带**——这决定了后面 mask token 也要加位置编码。

> 注意顺序：MAE 先对所有 patch 加位置编码，再做掩码（可见 patch 保留自己的 $u_i$）。两种顺序在数学上等价，实现上"先加后掩"更简洁。

## 2. 随机掩码

掩码集合从所有 $\rho N$ 大小的子集中均匀抽取：

$$
M \sim \mathrm{Uniform}\left\{ \mathcal{S} \subseteq \{0,\dots,N-1\} \;\middle|\; |\mathcal{S}| = \lfloor \rho N \rfloor \right\}, \qquad V = \{0,\dots,N-1\} \setminus M .
$$

每个 iteration 对每个样本**独立重采样**。两个后果：

1. **任务多样性**：同一图像在不同 epoch 呈现 $\binom{N}{\rho N}$ 种不同的"完形填空"，掩码本身构成指数级规模的增广——这是 MAE 无需强增广的形式化解释。
2. **期望意义上每个 patch 被监督的概率为 $\rho$**：长训练（1600 epoch）后，每个位置都被重构过足够多次。

实现层面的等价技巧（见 `code/mae_minimal.py` 的 `random_masking`）：对每个样本生成均匀噪声 $\varepsilon \in \mathbb{R}^N$，取 $\mathrm{argsort}(\varepsilon)$ 得到均匀随机置换 $\pi$，则 $V = \{\pi(0), \dots, \pi(K-1)\}$；逆置换 $\pi^{-1} = \mathrm{argsort}(\pi)$ 记为 `ids_restore`，供 decoder 恢复原顺序。

## 3. Encoder：只作用于可见 patch

$$
h^{(0)}_j = \tilde z_j \quad (j \in V), \qquad
h^{(\ell+1)} = \mathrm{Block}\big(h^{(\ell)}\big), \;\; \ell = 0, \dots, L-1, \qquad
h_j = h^{(L)}_j \;\; (j \in V),
$$

其中 $\mathrm{Block}$ 是标准 pre-norm ViT block：

$$
\mathrm{Block}(X) = X + \mathrm{MHA}(\mathrm{LN}(X)), \qquad
\mathrm{Block}(X) \leftarrow \mathrm{Block}(X) + \mathrm{MLP}(\mathrm{LN}(\mathrm{Block}(X))).
$$

关键结构性质：**encoder 的输入序列长度是 $K = (1-\rho)N$ 而非 $N$，且不含任何 mask token**。$\rho = 0.75, N = 196$ 时 $K = 49$。

## 4. Decoder：轻量、全 token、补全

decoder 的输入在**全部** $N$ 个位置上构造：

$$
c_i =
\begin{cases}
W_d\, h_i + b_d \in \mathbb{R}^{d'}, & i \in V \quad \text{（可见位置：encoder 输出的线性投影）}, \\[4pt]
\mathbf{e}_{\mathrm{mask}} \in \mathbb{R}^{d'}, & i \in M \quad \text{（被掩位置：共享可学习向量）},
\end{cases}
$$

$$
\hat c_i = c_i + v_i, \qquad i = 0, \dots, N-1 .
$$

$\mathbf{e}_{\mathrm{mask}}$ 对所有被掩位置**共享**，位置信息只来自 $v_i$——若不加 $v_i$，模型无法区分"要填的是哪个格子"。decoder 同样由若干（较少、较窄的）block 组成，输出逐位置回归：

$$
\hat y_i = \mathrm{Dec}_\phi\big(\hat c_0, \dots, \hat c_{N-1}\big)_i \in \mathbb{R}^{3p^2}, \qquad i \in M .
$$

结构上的**非对称性**一览：

| | encoder | decoder |
| --- | --- | --- |
| 处理的 token 数 | $K = (1-\rho)N$（49） | $N$（196） |
| 维度 / 深度 | $d = 1024$, $L = 24$（ViT-L） | $d' = 512$, 8 层 |
| 是否见 mask token | 否 | 是 |
| 下游是否保留 | 是（唯一交付物） | 否（预训练后丢弃） |

## 5. 重构目标与损失

先对目标 patch 做**逐 patch 归一化**（消除光照 / 对比度等低频分量）：

$$
\mu_i = \frac{1}{3p^2} \sum_{k=1}^{3p^2} x_{i,k}, \qquad
\sigma_i^2 = \frac{1}{3p^2} \sum_{k=1}^{3p^2} (x_{i,k} - \mu_i)^2, \qquad
\bar x_i = \frac{x_i - \mu_i \mathbf{1}}{\sqrt{\sigma_i^2 + \epsilon}} .
$$

损失为**只作用于被掩位置**的 MSE：

$$
\boxed{\;\;
\mathcal{L}(\theta, \phi; x, M)
= \frac{1}{\lvert M \rvert} \sum_{i \in M} \big\| \hat y_i - \bar x_i \big\|_2^2
\;\;}
$$

整体训练目标是对数据分布与掩码分布取期望：

$$
\min_{\theta, \phi}\; \mathbb{E}_{x \,\sim\, \mathcal{D},\; M \,\sim\, \mathrm{Uniform}} \big[ \mathcal{L}(\theta, \phi; x, M) \big].
$$

为什么只算 masked 位置：visible patch 的信息已经显式进入 encoder，重建它们退化为恒等映射，监督信号为零，只等价于给损失加噪声（BERT 的损失同样只算 masked token）。

## 6. 计算复杂度：可扩展性的来源

单层 Transformer 的 FLOPs 量级为 $O(N d^2 + N^2 d)$（MLP 项 + attention 项）。MAE 的 encoder 只处理 $K = (1-\rho)N$ 个 token：

$$
C_{\mathrm{enc}} = O\big( (1-\rho)\, N d^2 + (1-\rho)^2 N^2 d \big).
$$

取 $\rho = 0.75$：MLP 项降为 $1/4$，attention 项降为 $1/16$。decoder 虽处理全部 $N$ 个 token，但 $d' = d/2$、层数仅为 encoder 的 $1/3$，其代价相对可控。综合墙钟加速约 3× 量级（论文实测口径，具体倍数待核验）。

**本质**：普通 AE/DAE 的计算量与掩码率无关（decoder 总要看全部位置，encoder 若也看全部 token 则更贵）；MAE 把"计算量集中在 encoder"改成"encoder 只算可见部分"，于是**掩码率越高越快**——冗余性第一次转化为效率。

## 7. 与 BERT MLM 的公式级对照

$$
\mathcal{L}_{\mathrm{BERT}} = -\frac{1}{\lvert M \rvert} \sum_{i \in M} \log p_\theta\big( w_i \mid w_{\setminus M} \big), \qquad
\mathcal{L}_{\mathrm{MAE}} = \frac{1}{\lvert M \rvert} \sum_{i \in M} \big\| \hat y_i - \bar x_i \big\|^2 .
$$

| 维度 | BERT | MAE |
| --- | --- | --- |
| 掩码率 | 15% | 75% |
| 目标空间 | 离散词表（交叉熵） | 连续像素（MSE） |
| encoder 是否见 mask token | 见（`[`token`]` 进入双向编码） | 不见（非对称） |
| 损失位置 | 仅 masked | 仅 masked |
| 输入单元 | 语义 token（低冗余） | 视觉 patch（高冗余） |
| 预训练/微调错配 | 有（微调无 `[`mask`]`） | 无 |

最后一行常被忽视但很重要：MAE 的 encoder 在预训练和微调两个阶段看到的输入分布一致，这是非对称结构的一个"免费"副产品。

## 8. 小结

MAE 的数学骨架可以压缩成三行：

1. 均匀随机子集 $M$，encoder 只编码补集 $V$；
2. 共享 mask token + 位置编码补齐 $M$，轻量 decoder 逐位置回归；
3. 归一化像素上的 MSE，只在 $M$ 上平均。

下一章 `04-implementation.md` 把这三行逐字翻译成 PyTorch，并追踪每个张量的形状。
