"""Masked context + embedding-space prediction quality panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from visualizations.inference import forward_jepa_batch
from visualizations.style import PALETTE, save_figure

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _denorm(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().float()
    for c, (m, s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        img[c] = img[c] * s + m
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def _patch_rect(ax, row: int, col: int, patch_h: float, patch_w: float, color: str, alpha: float) -> None:
    ax.add_patch(
        Rectangle(
            (col * patch_w, row * patch_h),
            patch_w,
            patch_h,
            linewidth=0.6,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
        )
    )


def _masked_context_image(
    image: torch.Tensor,
    context_indices: torch.Tensor,
    grid_size: int,
) -> np.ndarray:
    """Gray out patches that are not in the context set."""
    img = _denorm(image).copy()
    h, w = img.shape[:2]
    ph, pw = h / grid_size, w / grid_size
    ctx = set(context_indices.detach().cpu().tolist())
    for idx in range(grid_size * grid_size):
        if idx in ctx:
            continue
        row, col = divmod(idx, grid_size)
        y0, x0 = int(row * ph), int(col * pw)
        y1, x1 = int((row + 1) * ph), int((col + 1) * pw)
        img[y0:y1, x0:x1] *= 0.25
    return img


def _similarity_heatmap(
    image: torch.Tensor,
    target_indices: torch.Tensor,
    pred_repr: torch.Tensor,
    target_repr: torch.Tensor,
    grid_size: int,
) -> np.ndarray:
    """Paint target patches by cosine similarity (pred vs teacher)."""
    img = _denorm(image).copy()
    h, w = img.shape[:2]
    ph, pw = h / grid_size, w / grid_size
    sims = F.cosine_similarity(pred_repr, target_repr, dim=-1).detach().cpu().numpy()
    tgt = target_indices.detach().cpu().tolist()
    for i, patch_idx in enumerate(tgt):
        row, col = divmod(patch_idx, grid_size)
        val = float(np.clip(sims[i], 0.0, 1.0))
        color = plt.cm.RdYlGn(val)[:3]
        y0, x0 = int(row * ph), int(col * pw)
        y1, x1 = int((row + 1) * ph), int((col + 1) * pw)
        patch = img[y0:y1, x0:x1]
        tint = np.array(color).reshape(1, 1, 3)
        img[y0:y1, x0:x1] = 0.55 * patch + 0.45 * tint
    return img


@torch.no_grad()
def plot_mask_reconstruction_comparison(
    baseline_model,
    looped_model,
    images: torch.Tensor,
    labels: torch.Tensor,
    masks,
    grid_size: int,
    class_names: list[str],
    output_stem: Path,
    device: torch.device | None = None,
) -> None:
    """Side-by-side baseline vs looped: original, masked context, prediction quality."""
    if device is None:
        device = next(baseline_model.parameters()).device
    n = min(images.shape[0], 6)
    fig, axes = plt.subplots(n, 6, figsize=(14, 2.4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "Original",
        "Masked context",
        "Pred quality (baseline)",
        "Original",
        "Masked context",
        "Pred quality (looped)",
    ]

    for row in range(n):
        image = images[row]
        label = int(labels[row].item())
        ctx = masks.context_indices[row]
        tgt = masks.target_indices[row]
        ph = image.shape[1] / grid_size
        pw = image.shape[2] / grid_size

        base_out = forward_jepa_batch(
            baseline_model,
            image.unsqueeze(0),
            ctx.unsqueeze(0),
            tgt.unsqueeze(0),
            device=device,
        )
        loop_out = forward_jepa_batch(
            looped_model,
            image.unsqueeze(0),
            ctx.unsqueeze(0),
            tgt.unsqueeze(0),
            device=device,
        )

        orig = _denorm(image)
        masked = _masked_context_image(image, ctx, grid_size)
        base_heat = _similarity_heatmap(image, tgt, base_out["pred_repr"][0], base_out["target_repr"][0], grid_size)
        loop_heat = _similarity_heatmap(image, tgt, loop_out["pred_repr"][0], loop_out["target_repr"][0], grid_size)

        panels = [orig, masked, base_heat, orig, masked, loop_heat]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(class_names[label], fontsize=9, rotation=0, labelpad=42, va="center")

        # Context / target outlines on first masked panel
        for col in (1, 4):
            ax = axes[row, col]
            ctx_set = set(ctx.detach().cpu().tolist())
            tgt_set = set(tgt.detach().cpu().tolist())
            for idx in range(grid_size * grid_size):
                r, c = divmod(idx, grid_size)
                if idx in tgt_set:
                    _patch_rect(ax, r, c, ph, pw, PALETTE["target"], 0.35)
                elif idx in ctx_set:
                    _patch_rect(ax, r, c, ph, pw, PALETTE["context"], 0.25)

    fig.suptitle(
        "Masked prediction in embedding space (greener target patches = higher cosine similarity)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, output_stem)
