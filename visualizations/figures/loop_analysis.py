"""Per-loop deep-dive figures for the looped predictor."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from visualizations.loop_collect import LoopSampleRecord
from visualizations.style import PALETTE, save_figure

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _denorm(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().float()
    for c, (m, s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        img[c] = img[c] * s + m
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def plot_exit_distribution_deep(
    records: list[LoopSampleRecord],
    output_stem: Path,
) -> None:
    """Histogram + CDF + mean exit-gate probability per loop index."""
    expected = np.array([r.expected_loops for r in records])
    loops = sorted(set(int(round(v)) for v in expected))
    counts = np.array([np.sum(np.round(expected) == k) for k in loops], dtype=float)
    probs = counts / counts.sum()
    cumulative = np.cumsum(probs)

    n_exit_loops = len(records[0].exit_probs) if records and records[0].exit_probs else 0
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    axes[0].bar(loops, probs, color=PALETTE["looped"], edgecolor="white", linewidth=0.6)
    axes[0].set_xlabel("expected loops (rounded)")
    axes[0].set_ylabel("fraction of samples")
    axes[0].set_title("Exit depth distribution")
    axes[0].set_xticks(loops)

    axes[1].plot(loops, cumulative, marker="o", color=PALETTE["looped"], linewidth=2)
    axes[1].set_xlabel("expected loops (rounded)")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].set_title("Cumulative distribution")
    axes[1].set_ylim(0, 1.05)

    if n_exit_loops:
        mean_exit = [float(np.mean([r.exit_probs[j] for r in records if r.exit_probs])) for j in range(n_exit_loops)]
        xs = np.arange(1, n_exit_loops + 1)
        axes[2].bar(xs, mean_exit, color=PALETTE["accent"], edgecolor="white")
        axes[2].set_xlabel("loop index")
        axes[2].set_ylabel("mean P(exit)")
        axes[2].set_title("Exit-gate activation by loop")
        axes[2].set_xticks(xs)
        axes[2].set_ylim(0, 1)
    else:
        axes[2].set_axis_off()

    mean_loops = float(expected.mean())
    fig.suptitle(f"Looped predictor exit behaviour (n={len(records)}, mean loops={mean_loops:.2f})", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_cosine_and_l1_by_loop(
    records: list[LoopSampleRecord],
    class_names: list[str],
    output_stem: Path,
) -> None:
    """Cosine + smooth-L1 vs loop with error bars; per-class cosine gain bars."""
    if not records or not records[0].cosine_by_loop:
        return
    n_loops = len(records[0].cosine_by_loop)
    xs = np.arange(1, n_loops + 1)

    cos_by_loop = np.array([[r.cosine_by_loop[j] for j in range(n_loops)] for r in records])
    l1_by_loop = np.array([[r.l1_by_loop[j] for j in range(n_loops)] for r in records])
    cos_mean = cos_by_loop.mean(axis=0)
    cos_std = cos_by_loop.std(axis=0)
    l1_mean = l1_by_loop.mean(axis=0)
    l1_std = l1_by_loop.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(xs, cos_mean, yerr=cos_std, marker="o", capsize=4, color=PALETTE["looped"], linewidth=2)
    axes[0].set_xlabel("predictor loop")
    axes[0].set_ylabel("cosine similarity (pred vs teacher target)")
    axes[0].set_xticks(xs)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Embedding alignment improves across loops")

    axes[1].errorbar(xs, l1_mean, yerr=l1_std, marker="s", capsize=4, color=PALETTE["target"], linewidth=2)
    axes[1].set_xlabel("predictor loop")
    axes[1].set_ylabel("smooth L1 (pred vs teacher target)")
    axes[1].set_xticks(xs)
    axes[1].set_title("Prediction error decreases across loops")

    fig.tight_layout()
    save_figure(fig, output_stem)

    if n_loops >= 2 and len(class_names) >= 1:
        gains_by_class: dict[int, list[float]] = {}
        for r in records:
            gains_by_class.setdefault(r.label, []).append(r.cosine_gain)
        labels_sorted = sorted(gains_by_class.keys())
        means = [float(np.mean(gains_by_class[c])) for c in labels_sorted]
        stds = [float(np.std(gains_by_class[c])) for c in labels_sorted]
        names = [class_names[c] if c < len(class_names) else str(c) for c in labels_sorted]

        fig2, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, capsize=3, color=PALETTE["looped"], edgecolor="white", alpha=0.9)
        ax.axhline(0, color="#333", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Δ cosine (final loop − loop 1)")
        ax.set_title("Per-class benefit from recurrence")
        fig2.tight_layout()
        save_figure(fig2, Path(str(output_stem) + "_per_class"))


def plot_loops_vs_difficulty(
    records: list[LoopSampleRecord],
    output_stem: Path,
) -> None:
    """Scatter: loop-1 cosine vs expected loops (does the gate use more loops when harder?)."""
    if not records or len(records[0].cosine_by_loop) < 1:
        return
    loop1_cos = np.array([r.cosine_by_loop[0] for r in records])
    expected = np.array([r.expected_loops for r in records])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(loop1_cos, expected, s=8, alpha=0.35, c=PALETTE["looped"], edgecolors="none")
    # Binned trend
    bins = np.linspace(loop1_cos.min(), loop1_cos.max(), 12)
    bin_idx = np.digitize(loop1_cos, bins) - 1
    bin_centers, bin_means = [], []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        bin_centers.append(0.5 * (bins[b] + bins[b + 1]))
        bin_means.append(float(expected[mask].mean()))
    if bin_centers:
        ax.plot(bin_centers, bin_means, color=PALETTE["target"], linewidth=2.5, label="binned mean")
        ax.legend(loc="lower left")

    ax.set_xlabel("loop-1 cosine similarity (lower = harder)")
    ax.set_ylabel("expected loops used")
    ax.set_title("Exit gate vs initial prediction difficulty")
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_early_vs_late_examples(
    records: list[LoopSampleRecord],
    images: list[torch.Tensor],
    class_names: list[str],
    output_stem: Path,
    n_each: int = 4,
) -> None:
    """Show images where the model exits early vs uses more loops."""
    early_idx = [i for i, r in enumerate(records) if r.loops_rounded <= 1]
    late_idx = [i for i, r in enumerate(records) if r.loops_rounded >= 2]

    def _pick(idxs: list[int], k: int) -> list[int]:
        if not idxs:
            return []
        # Prefer diverse classes
        chosen: list[int] = []
        seen: set[int] = set()
        for i in sorted(idxs, key=lambda j: records[j].cosine_gain):
            if records[i].label in seen:
                continue
            chosen.append(i)
            seen.add(records[i].label)
            if len(chosen) >= k:
                break
        if len(chosen) < k:
            for i in idxs:
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= k:
                    break
        return chosen[:k]

    early_pick = _pick(early_idx, n_each)
    late_pick = _pick(late_idx, n_each)
    rows = max(len(early_pick), len(late_pick), 1)

    fig, axes = plt.subplots(rows, 2, figsize=(7, 2.6 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    def _panel(ax, idx: int | None, title: str) -> None:
        ax.set_axis_off()
        if idx is None:
            ax.set_title(f"{title}\n(n/a)", fontsize=9)
            return
        r = records[idx]
        ax.imshow(_denorm(images[idx]))
        name = class_names[r.label] if r.label < len(class_names) else str(r.label)
        cos_str = " → ".join(f"{c:.2f}" for c in r.cosine_by_loop)
        ax.set_title(
            f"{title}\n{name} | loops≈{r.expected_loops:.2f}\ncos: {cos_str}",
            fontsize=8,
        )

    for row in range(rows):
        _panel(axes[row, 0], early_pick[row] if row < len(early_pick) else None, "Early exit (~1 loop)")
        _panel(axes[row, 1], late_pick[row] if row < len(late_pick) else None, "More loops (~2)")

    fig.suptitle("Exit gate selects depth per sample", fontsize=12, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_stem)


def generate_all_loop_analysis(
    records: list[LoopSampleRecord],
    images: list[torch.Tensor],
    class_names: list[str],
    out_dir: Path,
) -> None:
    """Write the full per-loop figure set to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_exit_distribution_deep(records, out_dir / "01_exit_distribution")
    plot_cosine_and_l1_by_loop(records, class_names, out_dir / "02_cosine_l1_by_loop")
    plot_loops_vs_difficulty(records, out_dir / "03_loops_vs_difficulty")
    plot_early_vs_late_examples(records, images, class_names, out_dir / "04_early_vs_late_examples")
