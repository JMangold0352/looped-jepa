"""Shared matplotlib styling for publication figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Muted, colorblind-friendly palette (Tableau-inspired).
PALETTE = {
    "baseline": "#4C78A8",
    "looped": "#F58518",
    "accent": "#54A24B",
    "target": "#E45756",
    "context": "#72B7B2",
    "neutral": "#B279A2",
    "grid": "#E5E5E5",
}

ABLATION_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#72B7B2",
    "#FF9DA6",
]

FIG_DPI = 300


def apply_style() -> None:
    """Apply a clean, publication-ready matplotlib theme."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Write PNG + PDF at publication DPI."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
