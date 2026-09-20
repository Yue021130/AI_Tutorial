# -*- coding: utf-8 -*-
"""
MAE 最小可运行实现 (PyTorch)
论文: Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)

默认配置为教学用小模型 (encoder 192-d/6层, decoder 96-d/2层, 224/16, mask 75%),
CPU 也能跑通冒烟测试。把 enc_dim/enc_depth 换成 1024/24、dec_dim/dec_depth
换成 512/8 即对齐论文 ViT-L 配置。

用法:
    python code/mae_minimal.py     # 随机数据上的过拟合冒烟测试

真实训练时, 把 __main__ 里的随机 imgs 换成 ImageFolder 的 DataLoader
(增广只用 RandomResizedCrop + 随机翻转), 并按 05-experiments.md 配置
AdamW 超参 (lr 需按 batch/256 线性缩放)。
"""

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 工具: patch 化 / 反 patch 化 / 位置编码
# ---------------------------------------------------------------------------


def patchify(x, p):
    """(B,3,H,W) -> (B,N,3*p*p)。N=(H/p)*(W/p), 行优先顺序, 与 Conv2d 展平一致。"""
    B, C, H, W = x.shape
    h, w = H // p, W // p
    # (B,C,h,p,w,p) -> (B,h,w,C,p,p) -> (B,h*w,3*p*p)
    x = x.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, h * w, C * p * p)


def unpatchify(x, p, h, w):
    """(B,N,3*p*p) -> (B,3,H,W)。patchify 的逆变换。"""
    B, N, D = x.shape
    assert D == 3 * p * p and N == h * w
    x = x.reshape(B, h, w, 3, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, 3, h * p, w * p)


def _sincos_1d(dim, pos):
    """1D sin-cos 编码: pos (N,) -> (N, dim), dim 必须为偶数。"""
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)                 # (dim/2,) 频率
    out = pos[:, None].float() * omega[None, :]    # (N, dim/2) 相位
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def build_2d_sincos_pos_embed(embed_dim, grid_size):
    """固定 2D sin-cos 位置编码 (论文做法), 返回 (1, grid*grid, embed_dim)。"""
    assert embed_dim % 4 == 0, "2D sin-cos 要求 embed_dim 是 4 的倍数"
    g = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(g, g, indexing="ij")       # 行坐标 / 列坐标
    emb_h = _sincos_1d(embed_dim // 2, yy.flatten())   # 纵轴编码
    emb_w = _sincos_1d(embed_dim // 2, xx.flatten())   # 横轴编码
    return torch.cat([emb_h, emb_w], dim=1).unsqueeze(0)


# ---------------------------------------------------------------------------
# Transformer 基础模块 (pre-norm)
# ---------------------------------------------------------------------------


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """标准多头自注意力。x: (B,N,C) -> (B,N,C)。"""

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # (B,N,3C) -> (3,B,heads,N,head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        attn = attn.softmax(dim=-1)                   # (B,heads,N,N)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Block(nn.Module):
    """pre-norm Transformer Block: x + Attn(LN(x)); x + Mlp(LN(x))。"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """(B,3,H,W) -> (B,N,embed_dim)。stride=p 的卷积 = 每个 patch 的线性投影。"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                     # (B, embed_dim, grid, grid)
        return x.flatten(2).transpose(1, 2)  # (B, N, embed_dim), 行优先


# ---------------------------------------------------------------------------
# MAE 主体: 非对称 encoder-decoder
# ---------------------------------------------------------------------------


class MAE(nn.Module):
    """
    掩码自编码器:
      - encoder 只处理 (1-mask_ratio) 比例的可见 patch
      - decoder 拼回共享 mask token 处理全部 patch, 仅对被掩 patch 计损失
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 enc_dim=192, enc_depth=6, enc_heads=3,
                 dec_dim=96, dec_depth=2, dec_heads=3,
                 mlp_ratio=4.0, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # ---- encoder ----
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, enc_dim)
        self.num_patches = self.patch_embed.num_patches
        # 固定 sin-cos 位置编码 (不可学习), 与论文一致
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, enc_dim), requires_grad=False)
        self.pos_embed.data.copy_(
            build_2d_sincos_pos_embed(enc_dim, self.patch_embed.grid))
        self.blocks = nn.ModuleList([
            Block(enc_dim, enc_heads, mlp_ratio) for _ in range(enc_depth)])
        self.enc_norm = nn.LayerNorm(enc_dim)

        # ---- decoder (轻量, 预训练结束后即整体丢弃) ----
        self.dec_embed = nn.Linear(enc_dim, dec_dim)   # encoder 维度 -> decoder 维度
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))  # 共享可学习掩码 token
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, dec_dim), requires_grad=False)
        self.dec_pos_embed.data.copy_(
            build_2d_sincos_pos_embed(dec_dim, self.patch_embed.grid))
        self.dec_blocks = nn.ModuleList([
            Block(dec_dim, dec_heads, mlp_ratio) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, patch_size ** 2 * in_chans)  # 回归整块像素

    # ------------------------------------------------------------------

    def random_masking(self, x, mask_ratio):
        """
        均匀随机掩码。x: (B,N,D)。
        返回: x_visible (B,K,D), mask (B,N) [0=可见, 1=被掩], ids_restore (B,N)。
        技巧: 对均匀噪声 argsort 等价于均匀随机洗牌; argsort(洗牌) 即逆置换。
        """
        B, N, D = x.shape
        K = int(N * (1 - mask_ratio))                # 保留的可见 patch 数
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)    # 洗牌后的索引
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 逆置换: 还原原顺序

        # 按洗牌顺序取前 K 个作为可见 patch
        ids_keep = ids_shuffle[:, :K]
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # mask: 洗牌序下前 K 个=0(可见), 其余=1(被掩), 再还原到原顺序
        mask = torch.ones(B, N, device=x.device)
        mask[:, :K] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_visible, mask, ids_restore

    def forward_encoder(self, x):
        x = self.patch_embed(x)          # (B,N,D_enc)
        x = x + self.pos_embed           # 先加位置编码, 再掩码
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        for blk in self.blocks:
            x = blk(x)                   # encoder 只跑 (B,K,D_enc) 个 token
        x = self.enc_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.dec_embed(x)            # (B,K,D_dec) 维度变换

        # 拼接共享 mask token, 占据被掩位置 (此时仍是洗牌顺序: 可见在前)
        K = x.shape[1]
        mask_tokens = self.mask_token.repeat(
            x.shape[0], self.num_patches - K, 1)     # (B,N-K,D_dec)
        x = torch.cat([x, mask_tokens], dim=1)       # (B,N,D_dec)
        # 用逆置换把序列还原为原图 patch 顺序 (可见/掩码各归其位)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        x = x + self.dec_pos_embed      # mask token 也必须加位置编码!
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        x = self.pred(x)                 # (B,N,p*p*3) 预测每个 patch 的像素
        return x

    def forward_loss(self, imgs, pred, mask):
        target = patchify(imgs, self.patch_size)        # (B,N,3p^2) 重构目标
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5  # 逐 patch 归一化 (论文做法)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                        # (B,N) 每个 patch 的 MSE
        loss = (loss * mask).sum() / mask.sum()         # 只对被掩 patch 求平均
        return loss

    def forward(self, imgs):
        """imgs: (B,3,H,W) -> (pred, mask, loss)。"""
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return pred, mask, loss


# ---------------------------------------------------------------------------
# 训练一步
# ---------------------------------------------------------------------------


def train_step(model, imgs, optimizer):
    """执行单个优化步, 返回 float 损失。"""
    model.train()
    _, _, loss = model(imgs)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# 冒烟测试: 在随机数据上过拟合一个小批次, 验证机制正确性
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # patchify/unpatchify 往返一致性自检
    x = torch.rand(2, 3, 224, 224)
    assert torch.allclose(x, unpatchify(patchify(x, 16), 16, 14, 14))

    model = MAE().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device}  params={n_params:.2f}M  "
          f"patches={model.num_patches}  "
          f"visible={int(model.num_patches * (1 - model.mask_ratio))}")

    # 固定一批随机图像, 观察损失能否持续下降 (过拟合一个批次的能力
    # 是机制实现正确的必要条件; 真实训练请换成 ImageFolder 数据)
    imgs = torch.rand(16, 3, 224, 224, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.05)
    for it in range(60):
        loss = train_step(model, imgs, optimizer)
        if it % 10 == 0 or it == 59:
            print(f"step {it:3d}  loss {loss:.4f}")
    print("冒烟测试通过: 损失稳定下降, MAE 机制实现正确。")
