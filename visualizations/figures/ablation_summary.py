"""Ablation summary bar charts from results/ablations/summary.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualizations.style import ABLATION_COLORS, save_figure


def plot_ablation_summary(summary_path: Path, output_stem: Path) -> None:
    """Grouped bar charts for tuned probe accuracy and feat_std across ablations."""
    data = json.loads(Path(summary_path).read_text())
    suites = data.get("suites", [])

    rows: list[tuple[str, str, float, float]] = []
    for suite in suites:
        suite_name = suite["suite"]
        for variant in suite.get("variants", []):
            if "top1_accuracy" not in variant:
                continue
            label = f"{suite_name}\n{variant['name']}"
            rows.append((label, variant["name"], variant["top1_accuracy"], variant.get("feat_std", 0.0)))

    if not rows:
        return

    labels = [r[0] for r in rows]
    accs = [r[2] for r in rows]
    stds = [r[3] for r in rows]
    x = np.arange(len(labels))
    colors = [ABLATION_COLORS[i % len(ABLATION_COLORS)] for i in range(len(labels))]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    bars0 = axes[0].bar(x, accs, color=colors, edgecolor="white", linewidth=0.6)
    axes[0].axhline(77.23, color="#333333", linestyle="--", linewidth=1.0, label="v3 baseline (77.23%)")
    axes[0].set_ylabel("tuned probe top-1 (%)")
    axes[0].set_ylim(70, max(accs) + 2)
    axes[0].legend(loc="lower right")
    for bar, val in zip(bars0, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.15, f"{val:.1f}", ha="center", fontsize=8)

    bars1 = axes[1].bar(x, stds, color=colors, edgecolor="white", linewidth=0.6)
    axes[1].set_ylabel("feat_std")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    for bar, val in zip(bars1, stds):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}", ha="center", fontsize=8)

    fig.suptitle("Looped predictor ablations (300-epoch v3 recipe)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_stem)
