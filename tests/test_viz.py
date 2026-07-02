from __future__ import annotations

import torch

from jepa.masking import IJEPAMaskCollator
from jepa.viz.plots import plot_mask_overlay


def test_plot_mask_overlay_writes_file(tmp_path) -> None:
    image = torch.randn(3, 32, 32)
    collator = IJEPAMaskCollator(grid_size=8, fixed_context_patches=32, fixed_target_patches=16)
    masks = collator(1)
    out = tmp_path / "mask.png"
    plot_mask_overlay(image, masks.context_indices[0], masks.target_indices[0], 8, out)
    assert out.exists()
    assert out.stat().st_size > 0
