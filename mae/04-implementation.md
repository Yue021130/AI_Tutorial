# 04 · 最小实现说明：`code/mae_minimal.py`

对照代码阅读本章。默认配置为教学用小模型（encoder 192 维 / 6 层，decoder 96 维 / 2 层，224/16，mask 75%），CPU 可跑冒烟测试；把 `enc_dim/enc_depth` 换成 `1024/24`、`dec_dim/dec_depth` 换成 `512/8` 即对齐论文 ViT-L 配置。

## 1. 端到端张量形状追踪

以 `B=16, 224/16, N=196, ρ=0.75 → K=49, enc_dim=192, dec_dim=96` 为例：

| 步骤 | 代码位置 | 张量 | 形状 |
| --- | --- | --- | --- |
| 输入 | `main` | `imgs` | `(16, 3, 224, 224)` |
| PatchEmbed（Conv2d k=s=16） | `PatchEmbed.forward` | `x` | `(16, 196, 192)` |
| 加位置编码 | `forward_encoder` | `x` | `(16, 196, 192)` |
| random_masking | `random_masking` | `x_visible` | `(16, 49, 192)` |
| | | `mask`（0=可见，1=被掩） | `(16, 196)` |
| | | `ids_restore` | `(16, 196)` |
| encoder blocks ×6 + norm | `forward_encoder` | `latent` | `(16, 49, 192)` |
| 维度投影 enc→dec | `forward_decoder` | `x` | `(16, 49, 96)` |
| 拼接 mask token | `forward_decoder` | `x` | `(16, 196, 96)`（洗牌序） |
| `gather(ids_restore)` 还原顺序 | `forward_decoder` | `x` | `(16, 196, 96)`（原图序） |
| + decoder 位置编码，blocks ×2，norm，pred | `forward_decoder` | `pred` | `(16, 196, 768)`，768 = 16×16×3 |
| patchify(imgs) 得目标 | `forward_loss` | `target` | `(16, 196, 768)` |
| 逐 patch 归一化 + MSE × mask | `forward_loss` | `loss` | 标量 |

## 2. 模块逐个说明

### 2.1 `patchify` / `unpatchify`
`(B,3,H,W) ↔ (B,N,3p²)` 的互逆变换。**顺序契约**：`reshape(B,C,h,p,w,p).permute(0,2,4,1,3,5)` 的行优先顺序必须与 `Conv2d` 输出 `flatten(2).transpose(1,2)` 的顺序、以及 2D sin-cos 位置编码 `meshgrid(indexing='ij')` 的展开顺序**完全一致**，否则掩码与位置会错位——三个函数共享同一顺序约定是本实现正确性的基石。`main` 里有往返一致性 assert。

### 2.2 `build_2d_sincos_pos_embed`
固定（不可学习）的 2D sin-cos 位置编码，与论文一致：维度对半分给纵横两轴，各做标准频率正余弦展开。要求 `embed_dim % 4 == 0`。注册为 `requires_grad=False` 的 `nn.Parameter`（官方仓库同款做法），也可用 `register_buffer`。

### 2.3 `PatchEmbed`
`nn.Conv2d(3, d, kernel_size=p, stride=p)` 一步完成"切 patch + 每 patch 线性投影"，比" unfold + Linear"更短且数值等价。输出 `(B, d, g, g) → flatten(2).transpose(1,2) → (B, N, d)`。

### 2.4 `random_masking`
均匀随机掩码的标准实现：
- `noise = torch.rand(B, N)` → `ids_shuffle = argsort(noise)` 得到**均匀随机置换**；
- 前 `K` 个为可见 patch，`torch.gather` 按下标抽取；
- `ids_restore = argsort(ids_shuffle)` 是逆置换，decoder 用它把"可见在前、掩码在后"的洗牌序列还原为原图顺序；
- `mask` 张量语义为 **0=可见、1=被掩**——这是全篇最容易弄反的地方，损失里 `(loss * mask).sum() / mask.sum()` 依赖该语义。

### 2.5 `Attention` / `Block` / `Mlp`
教科书式 pre-norm Transformer：`x + Attn(LN(x))` 与 `x + MLP(LN(x))`。注意力手写 `qkv → reshape → softmax(QKᵀ/√d) → V`，避免依赖 `nn.MultiheadAttention` 的封装，便于看清 encoder 序列长度从 196 变成 49 时计算发生在哪。

### 2.6 `MAE`
四个 `forward_*` 方法对应论文四步：
- `forward_encoder`：patch 化 → **加位置编码（先加后掩）** → 掩码 → 只对 49 个可见 token 跑 encoder；
- `forward_decoder`：enc→dec 维度投影 → **拼接共享 mask token（`nn.Parameter`，初始为 0）** → `gather(ids_restore)` 还原顺序 → **加 decoder 位置编码（mask token 也必须加！）** → 轻量 blocks → `pred` 线性头输出每 patch 的 768 维像素；
- `forward_loss`：`patchify` 目标 → 逐 patch 归一化（`var` 用 PyTorch 默认的无偏估计，与官方一致）→ MSE → **只在 mask=1 的位置求平均**；
- `forward`：串起来，返回 `(pred, mask, loss)`。

### 2.7 `train_step`
`model(imgs)` → `zero_grad` → `backward` → `step`。真实训练的完整优化器配置（lr 线性缩放、warmup、cosine、wd）见 `05-experiments.md`；冒烟测试里用固定小 lr 即可。

## 3. 运行方式与预期输出

```bash
python code/mae_minimal.py
```

预期（CPU 数十秒内）：
- 打印参数量（默认小模型约 3–4 M）、patch 数 196、可见 patch 数 49；
- patchify/unpatchify 往返 assert 通过；
- 固定随机批次上 60 步，**loss 单调明显下降**（小模型过拟合一个批次的能力 = 机制正确的必要条件）；
- 结束打印冒烟测试通过。

真实训练：把 `main` 里的随机 `imgs` 换成 `ImageFolder` + `RandomResizedCrop + flip` 的 DataLoader，按 `05-experiments.md` 的超参表配置优化器即可。

## 4. 与官方实现（facebookresearch/mae）的差异清单

| 项 | 本实现 | 官方 |
| --- | --- | --- |
| 模型规模 | 教学小模型 | ViT-B/L/H |
| cls token | 无 | 需核对（官方 encoder 是否含 cls token 待核验；微调用的 `models_vit.py` 含） |
| 正则化 | 无 drop path / dropout | ViT-L/H 带 drop path（待核验具体值） |
| 位置编码 | 固定 2D sin-cos | 一致 |
| 掩码实现 | 噪声 argsort | 一致（同款技巧） |
| 损失 | 逐 patch 归一化 MSE | 一致 |
| 优化器/调度 | 冒烟用固定 lr | AdamW + warmup + cosine（见 05） |

## 5. 自测题

1. 若把 `mask` 的 0/1 语义弄反，程序会报错吗？损失会变成什么样？（提示：不报错，但任务退化为重建可见 patch——恒等任务，loss 会异常低。）
2. 去掉 `x = x + self.dec_pos_embed` 中 mask token 位置上的编码，模型还能定位空位吗？
3. `ids_restore` 若误用 `ids_shuffle`，decoder 输出与损失会发生什么？
4. 为什么 `forward_loss` 里 `patchify(imgs)` 而不是复用 encoder 输入？（提示：encoder 输入是被位置编码污染过的 token，不是干净像素。）
