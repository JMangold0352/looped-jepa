"""Bar chart for scale comparison table (CIFAR vs EuroSAT vs TinyImageNet)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from visualizations.style import PALETTE, apply_style, save_figure

ROOT = PROJECT_ROOT
TABLE_JSON = ROOT / "results" / "scale" / "scale_table.json"
OUTPUT_STEM = ROOT / "visualizations" / "figures" / "06_scale_comparison"


def load_scale_rows(path: Path = TABLE_JSON) -> list[dict]:
    data = json.loads(path.read_text())
    return list(data.get("rows", []))


def plot_scale_bars(rows: list[dict] | None = None, output_stem: Path = OUTPUT_STEM) -> Path:
    apply_style()
    rows = rows or load_scale_rows()

    labels: list[str] = []
    baseline_vals: list[float] = []
    looped_vals: list[float] = []
    for row in rows:
        if row.get("baseline_top1") is None or row.get("looped_top1") is None:
            continue
        labels.append(row["dataset"])
        baseline_vals.append(float(row["baseline_top1"]))
        looped_vals.append(float(row["looped_top1"]))

    if not labels:
        raise RuntimeError("No rows with both baseline and looped top-1 in scale_table.json")

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, baseline_vals, width, label="baseline", color=PALETTE["baseline"])
    ax.bar(x + width / 2, looped_vals, width, label="looped", color=PALETTE["looped"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("tuned linear probe top-1 (%)")
    ax.set_title("Recurrence vs baseline across scale / domain")
    ax.legend()
    ax.set_ylim(0, max(baseline_vals + looped_vals) * 1.12)

    for i, (b, l) in enumerate(zip(baseline_vals, looped_vals)):
        delta = l - b
        ax.annotate(
            f"{delta:+.1f} pp",
            xy=(x[i], max(b, l) + 1.5),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, output_stem)
    return output_stem.with_suffix(".png")


if __name__ == "__main__":
    out = plot_scale_bars()
    print(f"Wrote {out}")
