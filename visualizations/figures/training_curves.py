"""Training metrics and exit-loop distribution figures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from jepa.viz.plots import load_metrics_jsonl
from visualizations.style import PALETTE, save_figure


def _probe_series(records: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    """Probe metrics are logged at steps; derive a pseudo-epoch from step if needed."""
    xs, ys = [], []
    steps_per_epoch = None
    epoch_losses = [r for r in records if "epoch" in r and "train/epoch_loss" in r]
    if len(epoch_losses) >= 2:
        steps_per_epoch = epoch_losses[1]["step"] - epoch_losses[0]["step"]
    for r in records:
        if key not in r:
            continue
        if "epoch" in r:
            xs.append(int(r["epoch"]))
        elif steps_per_epoch:
            xs.append(max(1, int(round(r["step"] / steps_per_epoch))))
        else:
            xs.append(int(r["step"]))
        ys.append(float(r[key]))
    return xs, ys


def _epoch_series(records: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in records:
        if key in r and "epoch" in r:
            xs.append(int(r["epoch"]))
            ys.append(float(r[key]))
    return xs, ys


def _step_series(records: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in records:
        if key in r:
            xs.append(int(r.get("step", len(xs))))
            ys.append(float(r[key]))
    return xs, ys


def plot_training_evaluation_curves(
    baseline_metrics: Path,
    looped_metrics: Path,
    output_stem: Path,
) -> None:
    """Loss, probe accuracy, and feat_std over training for baseline vs looped."""
    base_records = load_metrics_jsonl(baseline_metrics)
    loop_records = load_metrics_jsonl(looped_metrics)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for records, label, color in (
        (base_records, "v3 baseline", PALETTE["baseline"]),
        (loop_records, "v3 looped", PALETTE["looped"]),
    ):
        ex, ey = _epoch_series(records, "train/epoch_loss")
        if ex:
            axes[0].plot(ex, ey, label=label, color=color)
        px, py = _probe_series(records, "eval/probe_top1")
        if px:
            axes[1].plot(px, py, label=label, color=color, marker="o", markersize=3)
        fx, fy = _probe_series(records, "eval/feat_std")
        if fx:
            axes[2].plot(fx, fy, label=label, color=color, marker="o", markersize=3)

    axes[0].set_ylabel("train loss")
    axes[1].set_ylabel("probe top-1 (%)")
    axes[2].set_ylabel("feat_std")
    axes[2].set_xlabel("epoch")
    for ax in axes:
        ax.legend(loc="best")
    fig.suptitle("Training & evaluation curves", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_exit_loop_distribution(
    loop_usage: dict[str, Any],
    output_stem: Path,
) -> None:
    """Histogram + cumulative distribution of expected loop counts."""
    hist = loop_usage.get("loops_used_histogram", {})
    if not hist:
        return

    loops = sorted(int(k) for k in hist)
    counts = np.array([hist[str(k)] if str(k) in hist else hist[k] for k in loops], dtype=float)
    probs = counts / counts.sum()
    cumulative = np.cumsum(probs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(loops, probs, color=PALETTE["looped"], edgecolor="white", linewidth=0.6)
    axes[0].set_xlabel("loops used (rounded)")
    axes[0].set_ylabel("fraction of samples")
    axes[0].set_title("Exit loop distribution")
    axes[0].set_xticks(loops)

    axes[1].plot(loops, cumulative, marker="o", color=PALETTE["looped"])
    axes[1].set_xlabel("loops used (rounded)")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].set_title("Cumulative exit distribution")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(loops)

    mean_loops = loop_usage.get("mean_loops_used", 0)
    fig.suptitle(f"Looped predictor exit behaviour (mean loops = {mean_loops:.2f})", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_expected_loops_training(
    looped_metrics: Path,
    output_stem: Path,
) -> None:
    """Training-time expected loop depth from logged metrics."""
    records = load_metrics_jsonl(looped_metrics)
    xs, ys = _step_series(records, "train/expected_loops")
    if not xs:
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(xs, ys, color=PALETTE["looped"], linewidth=1.2)
    ax.set_xlabel("training step")
    ax.set_ylabel("expected loops")
    ax.set_title("Exit gate: expected loops during training")
    fig.tight_layout()
    save_figure(fig, output_stem)
