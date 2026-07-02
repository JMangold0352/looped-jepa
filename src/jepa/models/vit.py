from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth: randomly zero whole residual branches per sample."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # One mask value per sample; broadcasts over the remaining dims.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def default_norm_factory(dim: int) -> nn.Module:
    return nn.LayerNorm(dim)


def rms_norm_factory(dim: int) -> nn.Module:
    return RMSNorm(dim)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4)
        act = nn.GELU() if activation == "gelu" else nn.SiLU()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        hidden_dim = int(math.ceil(hidden_dim / 64) * 64)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        x = self.w_down(gate * up)
        return self.drop(x)


class RotaryEmbedding2D(nn.Module):
    """2D RoPE for patch tokens arranged on a square grid."""

    def __init__(self, dim: int, grid_size: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("RoPE head dim must be divisible by 4")
        half = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.grid_size = grid_size

    def _rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (B, H, N, D), positions: (N,)
        freqs = torch.einsum("n,d->nd", positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + rot * sin

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, n, d = q.shape
        side = self.grid_size
        if n != side * side:
            return q, k

        rows = torch.arange(side, device=q.device).repeat_interleave(side)
        cols = torch.arange(side, device=q.device).repeat(side)
        q_y, q_x = q.chunk(2, dim=-1)
        k_y, k_x = k.chunk(2, dim=-1)
        q_y = self._rotate(q_y, rows)
        q_x = self._rotate(q_x, cols)
        k_y = self._rotate(k_y, rows)
        k_x = self._rotate(k_x, cols)
        return torch.cat([q_y, q_x], dim=-1), torch.cat([k_y, k_x], dim=-1)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope: RotaryEmbedding2D | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            q, k = self.rope.apply(q, k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ViT block with swappable norm and optional sandwich (pre-attn + pre-ffn) norms."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_factory: Callable[[int], nn.Module] = default_norm_factory,
        sandwich_norm: bool = False,
        ffn_type: str = "mlp",
        rope: RotaryEmbedding2D | None = None,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.sandwich_norm = sandwich_norm
        hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_factory(dim)
        self.norm2 = norm_factory(dim)
        # Sandwich layout adds a norm after each sub-layer inside the residual path.
        self.norm_attn_out = norm_factory(dim) if sandwich_norm else None
        self.norm_ffn_out = norm_factory(dim) if sandwich_norm else None

        self.attn = Attention(dim, num_heads=num_heads, proj_drop=dropout, rope=rope)
        if ffn_type == "swiglu":
            self.mlp = SwiGLU(dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            self.mlp = Mlp(dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.norm_attn_out is not None:
            x = self.norm_attn_out(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.norm_ffn_out is not None:
            x = self.norm_ffn_out(x)
        return x


class BlockStack(nn.Module):
    """Reusable stack of transformer blocks (used by predictor for looped recurrence)."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class VICRegProjector(nn.Module):
    """MLP projector for the variance/covariance regularizer."""

    def __init__(self, in_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def vicreg_variance_covariance(z: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance hinge + off-diagonal covariance penalty (no invariance term)."""
    std = torch.sqrt(z.var(dim=0) + eps)
    variance_loss = torch.relu(1.0 - std).mean()

    z = z - z.mean(dim=0)
    n, d = z.shape
    cov = (z.T @ z) / max(1, n - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    covariance_loss = off_diag.pow(2).sum() / d
    return variance_loss, covariance_loss


def build_pos_embed(num_patches: int, embed_dim: int) -> nn.Parameter:
    return nn.Parameter(torch.zeros(1, num_patches, embed_dim))


def init_pos_embed(pos_embed: nn.Parameter) -> None:
    nn.init.trunc_normal_(pos_embed, std=0.02)
