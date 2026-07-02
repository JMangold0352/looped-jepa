"""Smoke test for the visualization pipeline."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_generate_all_figures_fast() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "visualizations/generate_all_figures.py"), "--fast", "--embed-method", "pca"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    out_dir = root / "visualizations/figures"
    assert (out_dir / "01_mask_reconstruction.png").exists()
    assert (out_dir / "05_ablation_summary.pdf").exists()
    loop_dir = root / "visualizations/loop_analysis"
    assert (loop_dir / "01_exit_distribution.png").exists()
    assert (loop_dir / "summary.json").exists()
