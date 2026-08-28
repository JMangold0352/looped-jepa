"""Metrics and panels for notebooks/visualize_latents.ipynb (not full figure regen)."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from visualizations.style import PALETTE, save_figure


@dataclass
class SeparationMetrics:
    silhouette: float
    nearest_centroid_acc: float


def _to_numpy(feats: torch.Tensor) -> np.ndarray:
    return feats.detach().cpu().numpy()


def separation_metrics(features: torch.Tensor, labels: torch.Tensor) -> SeparationMetrics:
    """Silhouette score and nearest-class-centroid accuracy on frozen features."""
    x = _to_numpy(features)
    y = _to_numpy(labels).astype(int)
    from sklearn.metrics import silhouette_score

    sil = float(silhouette_score(x, y, metric="euclidean"))
    classes = np.unique(y)
    centroids = np.stack([x[y == c].mean(axis=0) for c in classes])
    dists = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
    pred = classes[dists.argmin(axis=1)]
    acc = float((pred == y).mean())
    return SeparationMetrics(silhouette=sil, nearest_centroid_acc=acc)


def nearest_centroid_confusion_pairs(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    top_k: int = 5,
) -> list[tuple[str, str, float, int]]:
    """Return worst (true, pred) class pairs by off-diagonal count from data."""
    x = _to_numpy(features)
    y = _to_numpy(labels).astype(int)
    classes = np.unique(y)
    centroids = {c: x[y == c].mean(axis=0) for c in classes}
    preds = []
    for row in x:
        dists = {c: np.linalg.norm(row - centroids[c]) for c in classes}
        preds.append(min(dists, key=dists.get))
    preds = np.array(preds)

    n_cls = int(classes.max()) + 1
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y, preds):
        cm[t, p] += 1

    pairs: list[tuple[str, str, float, int]] = []
    for i in range(n_cls):
        for j in range(n_cls):
            if i == j:
                continue
            count = int(cm[i, j])
            if count == 0:
                continue
            denom = max(1, int((y == i).sum()))
            rate = count / denom
            ti = class_names[i] if i < len(class_names) else str(i)
            tj = class_names[j] if j < len(class_names) else str(j)
            pairs.append((ti, tj, rate, count))
    pairs.sort(key=lambda t: (t[2], t[3]), reverse=True)
    return pairs[:top_k]


def compare_separation(
    baseline_feats: torch.Tensor,
    looped_feats: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, SeparationMetrics]:
    return {
        "baseline": separation_metrics(baseline_feats, labels),
        "looped": separation_metrics(looped_feats, labels),
    }


def feature_norms(features: torch.Tensor) -> np.ndarray:
    return torch.linalg.norm(features, dim=1).detach().cpu().numpy()


def plot_feat_std_panel(
    feat_dict: dict[str, torch.Tensor],
    output_stem=None,
    *,
    reference_feat_std: dict[str, float] | None = None,
) -> plt.Figure:
    """Distribution of L2 feature norms; addresses tight scaling vs collapse."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = {
        "baseline_v3": PALETTE["baseline"],
        "looped_v3": PALETTE["looped"],
        "sandwich_rmsnorm": PALETTE.get("accent", "#e67e22"),
    }
    names = list(feat_dict.keys())
    for name, feats in feat_dict.items():
        norms = feature_norms(feats)
        color = colors.get(name, "#555")
        axes[0].hist(norms, bins=40, alpha=0.55, label=name, color=color, density=True)
        axes[1].boxplot(
            [norms],
            positions=[names.index(name) + 1],
            widths=0.6,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.55},
            medianprops={"color": "black"},
        )

    axes[0].set_xlabel("L2 norm (mean-pooled encoder feature)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Feature norm distribution (raw, unstandardized)")
    axes[0].legend(fontsize=8)

    axes[1].set_xticks(range(1, len(names) + 1))
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("L2 norm")
    axes[1].set_title("Norm spread (lower spread ≠ collapse if probe rises)")

    if reference_feat_std:
        txt = "Registry tuned-probe feat_std (train-split std): " + ", ".join(
            f"{k}={v:.4f}" for k, v in reference_feat_std.items()
        )
        fig.suptitle(txt, fontsize=10, y=1.02)

    fig.tight_layout()
    if output_stem is not None:
        save_figure(fig, output_stem)
    return fig


def per_class_probe_disagreements(
    baseline_feats: torch.Tensor,
    looped_feats: torch.Tensor,
    labels: torch.Tensor,
    baseline_head,
    looped_head,
    device: torch.device,
    *,
    n_each: int = 6,
) -> dict[str, list[int]]:
    """Indices where one probe is correct and the other is wrong."""
    with torch.no_grad():
        b_logits = baseline_head(baseline_feats.to(device)).argmax(1).cpu()
        l_logits = looped_head(looped_feats.to(device)).argmax(1).cpu()
    y = labels.cpu()
    b_ok = b_logits == y
    l_ok = l_logits == y
    return {
        "looped_wrong_baseline_right": (l_ok.logical_not() & b_ok).nonzero(as_tuple=True)[0].tolist()[:n_each],
        "baseline_wrong_looped_right": (b_ok.logical_not() & l_ok).nonzero(as_tuple=True)[0].tolist()[:n_each],
    }


def plot_disagreement_tiles(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    indices: list[int],
    title: str,
    baseline_preds: torch.Tensor | None = None,
    looped_preds: torch.Tensor | None = None,
    output_stem=None,
) -> plt.Figure:
    """Qualitative grid: image + true class + optional probe argmax labels."""
    from visualizations.figures.loop_analysis import _denorm

    n = len(indices)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No disagreement examples in this slice", ha="center")
        ax.axis("off")
        return fig

    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.atleast_2d(axes)
    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax, idx in zip(axes.flat, indices):
        img = _denorm(images[idx])
        ax.imshow(img)
        true_name = class_names[int(labels[idx])]
        caption = f"true: {true_name}"
        if baseline_preds is not None and looped_preds is not None:
            b_name = class_names[int(baseline_preds[idx])]
            l_name = class_names[int(looped_preds[idx])]
            caption += f"\nb: {b_name}  l: {l_name}"
        ax.set_title(caption, fontsize=7)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if output_stem is not None:
        save_figure(fig, output_stem)
    return fig
