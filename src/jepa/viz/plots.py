from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_metrics_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a ``metrics.jsonl`` run log into a list of records."""
    records: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_metrics(
    metrics_path: str | Path,
    output_path: str | Path,
    keys: tuple[str, ...] = ("train/loss", "eval/probe_top1", "eval/feat_std"),
) -> Figure:
    """Plot selected scalar metrics over training steps."""
    records = load_metrics_jsonl(metrics_path)

    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 2.8 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        xs, ys = [], []
        for r in records:
            if key in r:
                xs.append(r["step"])
                ys.append(r[key])
        ax.plot(xs, ys, linewidth=1.5)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("step")
    fig.suptitle(Path(metrics_path).parent.name, fontsize=12)
    fig.tight_layout()

    out = Path(output_path)
    _ensure_parent(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_probe_lr_sweep(
    results_by_lr: dict[float, float],
    output_path: str | Path,
    title: str = "Tuned linear probe LR sweep",
) -> Figure:
    """Bar chart of validation top-1 vs probe learning rate."""
    lrs = sorted(results_by_lr.keys())
    accs = [results_by_lr[lr] for lr in lrs]
    labels = [f"{lr:.0e}" for lr in lrs]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, accs, color="#4C72B0")
    ax.set_xlabel("probe learning rate")
    ax.set_ylabel("val top-1 (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.5, f"{acc:.1f}", ha="center", fontsize=9)
    fig.tight_layout()

    out = Path(output_path)
    _ensure_parent(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_mask_overlay(
    image: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    grid_size: int,
    output_path: str | Path,
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
) -> Figure:
    """Draw context (green) and target (red) patch masks on a CIFAR-scale image."""
    img = image.detach().cpu().float()
    if img.ndim == 3:
        for c, m, s in zip(range(3), mean, std):
            img[c] = img[c] * s + m
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    else:
        img = img.numpy()

    h, w = img.shape[:2]
    patch_h = h / grid_size
    patch_w = w / grid_size

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.set_axis_off()

    ctx = set(context_indices.detach().cpu().tolist())
    tgt = set(target_indices.detach().cpu().tolist())
    for idx in range(grid_size * grid_size):
        row, col = divmod(idx, grid_size)
        color = None
        alpha = 0.0
        if idx in tgt:
            color = "#E45756"
            alpha = 0.45
        elif idx in ctx:
            color = "#54A24B"
            alpha = 0.35
        if color:
            rect = Rectangle(
                (col * patch_w, row * patch_h),
                patch_w,
                patch_h,
                linewidth=0.8,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)

    fig.tight_layout()
    out = Path(output_path)
    _ensure_parent(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_embedding_pca(
    features: torch.Tensor,
    labels: torch.Tensor,
    output_path: str | Path,
    title: str = "Encoder embeddings (PCA)",
    max_points: int = 2000,
) -> Figure:
    """2D PCA scatter of frozen encoder features colored by class."""
    x = features.detach().cpu().float().numpy()
    y = labels.detach().cpu().numpy()
    if x.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(x.shape[0], max_points, replace=False)
        x, y = x[idx], y[idx]

    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:2].T

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=y, cmap="tab10", s=8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="class")
    fig.tight_layout()

    out = Path(output_path)
    _ensure_parent(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
