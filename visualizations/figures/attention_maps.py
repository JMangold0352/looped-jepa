"""Predictor attention visualizations for baseline and looped models."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from visualizations.inference import predictor_attention_maps
from visualizations.style import PALETTE, save_figure

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _denorm(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().float()
    for c, (m, s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        img[c] = img[c] * s + m
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def _attention_on_grid(
    attn_tgt_ctx: torch.Tensor,
    target_indices: torch.Tensor,
    grid_size: int,
) -> np.ndarray:
    """Map per-target context attention back to an 8×8 patch grid."""
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    tgt = target_indices.detach().cpu().tolist()
    # attn_tgt_ctx: (n_tgt, n_ctx), mean attention onto context per target token.
    weights = attn_tgt_ctx.detach().cpu().numpy()
    for j, patch_idx in enumerate(tgt):
        row, col = divmod(patch_idx, grid_size)
        grid[row, col] = float(weights[j].mean())
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid


@torch.no_grad()
def plot_attention_comparison(
    baseline_model,
    looped_model,
    images: torch.Tensor,
    labels: torch.Tensor,
    masks,
    grid_size: int,
    class_names: list[str],
    output_stem: Path,
    n_examples: int = 3,
    device: torch.device | None = None,
) -> None:
    """Baseline single-pass vs looped per-loop attention on context patches."""
    if device is None:
        device = next(baseline_model.parameters()).device
    images = images.to(device)
    n = min(n_examples, images.shape[0])
    ctx0 = masks.context_indices[:1]
    tgt0 = masks.target_indices[:1]
    looped_info = predictor_attention_maps(
        looped_model,
        images[:1],
        torch.stack(ctx0).to(device),
        torch.stack(tgt0).to(device),
    )
    n_loop_cols = len(looped_info["maps"])

    fig, axes = plt.subplots(n, 2 + n_loop_cols, figsize=(2.2 * (2 + n_loop_cols), 2.4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for row in range(n):
        image = images[row]
        ctx = torch.stack(masks.context_indices[row : row + 1]).to(device)
        tgt = torch.stack(masks.target_indices[row : row + 1]).to(device)
        label = int(labels[row].item())

        axes[row, 0].imshow(_denorm(image))
        axes[row, 0].set_axis_off()
        if row == 0:
            axes[row, 0].set_title("Input", fontsize=10)
        axes[row, 0].set_ylabel(class_names[label], fontsize=9, rotation=0, labelpad=36, va="center")

        base_info = predictor_attention_maps(baseline_model, image.unsqueeze(0), ctx, tgt)
        base_grid = _attention_on_grid(base_info["maps"][0][0], tgt[0], grid_size)
        im = axes[row, 1].imshow(base_grid, cmap="magma", vmin=0, vmax=1)
        axes[row, 1].set_axis_off()
        if row == 0:
            axes[row, 1].set_title("Baseline (1 pass)", fontsize=10)

        loop_info = predictor_attention_maps(looped_model, image.unsqueeze(0), ctx, tgt)
        for loop_i, attn_map in enumerate(loop_info["maps"]):
            grid = _attention_on_grid(attn_map[0], tgt[0], grid_size)
            ax = axes[row, 2 + loop_i]
            ax.imshow(grid, cmap="magma", vmin=0, vmax=1)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"Looped pass {loop_i + 1}", fontsize=10)

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="ctx attention (norm)")
    fig.suptitle("Target-patch attention to context (predictor, last block)", fontsize=12, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_stem)
