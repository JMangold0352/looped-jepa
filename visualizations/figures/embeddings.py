"""Encoder embedding and per-loop cosine similarity figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from visualizations.inference import predict_per_loop_cosine
from visualizations.style import PALETTE, save_figure


def _project_2d(features: np.ndarray, method: str = "tsne", seed: int = 0) -> np.ndarray:
    """2D projection via t-SNE (preferred) or PCA fallback."""
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            perplexity = min(30, max(5, features.shape[0] // 50))
            return TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=seed,
            ).fit_transform(features)
        except ImportError:
            pass

    x = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


@torch.no_grad()
def plot_embedding_comparison(
    baseline_feats: torch.Tensor,
    looped_feats: torch.Tensor,
    labels: torch.Tensor,
    output_stem: Path,
    max_points: int = 2000,
    method: str = "tsne",
) -> None:
    """Side-by-side 2D embedding projections for baseline vs looped encoders."""
    rng = np.random.default_rng(0)
    n = baseline_feats.shape[0]
    if n > max_points:
        idx = rng.choice(n, max_points, replace=False)
        baseline_feats = baseline_feats[idx]
        looped_feats = looped_feats[idx]
        labels = labels[idx]

    y = labels.detach().cpu().numpy()
    base_proj = _project_2d(baseline_feats.detach().cpu().numpy(), method=method)
    loop_proj = _project_2d(looped_feats.detach().cpu().numpy(), method=method)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, proj, title, color in zip(
        axes,
        [base_proj, loop_proj],
        ["v3 baseline encoder", "v3 + looped predictor encoder"],
        [PALETTE["baseline"], PALETTE["looped"]],
    ):
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=y, cmap="tab10", s=10, alpha=0.75, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="class")
    proj_name = "t-SNE" if method == "tsne" else "PCA"
    fig.suptitle(f"Frozen encoder features ({proj_name})", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_stem)


@torch.no_grad()
def plot_per_loop_cosine(
    looped_model,
    val_loader,
    collator,
    device: torch.device,
    output_stem: Path,
    max_batches: int = 32,
) -> None:
    """Line plot: mean cosine similarity vs predictor loop index."""
    sims_by_loop: list[list[float]] = []

    for batch_idx, (images, _) in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        masks = collator(images.shape[0])
        ctx = torch.stack(masks.context_indices).to(device)
        tgt = torch.stack(masks.target_indices).to(device)
        batch_sims = predict_per_loop_cosine(looped_model, images, ctx, tgt)
        max_loops = len(batch_sims)
        for loop_i in range(max_loops):
            while len(sims_by_loop) <= loop_i:
                sims_by_loop.append([])
            sims_by_loop[loop_i].append(batch_sims[loop_i])

    means = [float(np.mean(vals)) for vals in sims_by_loop]
    stds = [float(np.std(vals)) for vals in sims_by_loop]
    xs = np.arange(1, len(means) + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=4, color=PALETTE["looped"], label="looped predictor")
    ax.set_xlabel("predictor loop")
    ax.set_ylabel("cosine similarity (pred vs teacher target)")
    ax.set_xticks(xs)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Target embedding prediction quality across loops")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, output_stem)
