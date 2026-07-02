"""Rendering helpers: turn tensors / stats into clean PIL images for the demo UI."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DISPLAY_PX = 320
_BASELINE_C = "#4C78A8"
_LOOPED_C = "#F58518"


def denorm(image: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalized tensor -> HWC float array in [0,1]."""
    img = image.detach().cpu().float().clone()
    for c, (m, s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        img[c] = img[c] * s + m
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def to_pil(arr: np.ndarray, size: int = DISPLAY_PX) -> Image.Image:
    """HWC float [0,1] -> upscaled PIL image (nearest, crisp patch edges)."""
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    return img.resize((size, size), Image.NEAREST)


def masked_context(image: torch.Tensor, ctx_indices: torch.Tensor, grid: int) -> Image.Image:
    """Dim every patch that is not part of the visible context."""
    img = denorm(image).copy()
    h, w = img.shape[:2]
    ph, pw = h / grid, w / grid
    ctx = set(int(i) for i in ctx_indices.detach().cpu().tolist())
    for idx in range(grid * grid):
        if idx in ctx:
            continue
        r, c = divmod(idx, grid)
        y0, x0 = int(r * ph), int(c * pw)
        y1, x1 = int((r + 1) * ph), int((c + 1) * pw)
        img[y0:y1, x0:x1] *= 0.18
    return to_pil(img)


def prediction_quality(
    image: torch.Tensor,
    tgt_indices: torch.Tensor,
    cos_per_patch: np.ndarray,
    grid: int,
) -> Image.Image:
    """Tint each target patch by predicted-vs-teacher cosine (red->green)."""
    img = denorm(image).copy()
    h, w = img.shape[:2]
    ph, pw = h / grid, w / grid
    cmap = matplotlib.colormaps["RdYlGn"]
    for i, patch_idx in enumerate(tgt_indices.detach().cpu().tolist()):
        r, c = divmod(int(patch_idx), grid)
        val = float(np.clip(cos_per_patch[i], 0.0, 1.0))
        color = np.array(cmap(val)[:3]).reshape(1, 1, 3)
        y0, x0 = int(r * ph), int(c * pw)
        y1, x1 = int((r + 1) * ph), int((c + 1) * pw)
        img[y0:y1, x0:x1] = 0.5 * img[y0:y1, x0:x1] + 0.5 * color
    return to_pil(img)


def attention_overlay(image: torch.Tensor, attn_grid: np.ndarray, grid: int) -> Image.Image:
    """Overlay a normalized attention heatmap (magma) on a desaturated image."""
    base = denorm(image)
    gray = base.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.6
    a = attn_grid.astype(np.float32)
    if a.max() > a.min():
        a = (a - a.min()) / (a.max() - a.min())
    a_img = np.array(Image.fromarray((a * 255).astype(np.uint8)).resize((base.shape[1], base.shape[0]), Image.BILINEAR)) / 255.0
    heat = matplotlib.colormaps["magma"](a_img)[:, :, :3]
    alpha = (0.35 + 0.55 * a_img)[..., None]
    out = (1 - alpha) * gray + alpha * heat
    return to_pil(out)


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def cosine_curve(baseline_mean: float, looped_per_loop: list[float]) -> Image.Image:
    """Line chart: looped per-loop cosine vs the single-pass baseline reference."""
    loops = list(range(1, len(looped_per_loop) + 1))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(loops, looped_per_loop, "-o", color=_LOOPED_C, lw=2.2, ms=7, label="Looped (per loop)")
    ax.axhline(baseline_mean, ls="--", color=_BASELINE_C, lw=2.0, label="Baseline (1 pass)")
    ax.set_xlabel("Predictor loop")
    ax.set_ylabel("Cosine similarity to teacher")
    ax.set_title("Latent prediction quality vs refinement depth")
    ax.set_xticks(loops)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _fig_to_pil(fig)


def exit_gate_bars(exit_probs: list[float]) -> Image.Image:
    """Bar chart of per-loop exit probability from the learned gate."""
    loops = [f"loop {i + 1}" for i in range(len(exit_probs))]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(loops, exit_probs, color=_LOOPED_C, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(exit at loop)")
    ax.set_title("Learned exit-gate behavior")
    for i, p in enumerate(exit_probs):
        ax.text(i, min(p + 0.03, 0.97), f"{p:.2f}", ha="center", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return _fig_to_pil(fig)


def placeholder(text: str, size: int = DISPLAY_PX) -> Image.Image:
    fig, ax = plt.subplots(figsize=(size / 100, size / 100))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, color="#666", wrap=True)
    ax.set_axis_off()
    return _fig_to_pil(fig)
