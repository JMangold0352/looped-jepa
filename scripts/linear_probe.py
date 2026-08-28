#!/usr/bin/env python3
"""Tuned linear probe on a frozen JEPA encoder (official evaluation metric).

Usage::

    python scripts/linear_probe.py \\
        --config configs/image_jepa_cifar10_v3.yaml \\
        --checkpoint checkpoints/baseline_v3/latest.pt

    looped-jepa-probe --config configs/image_jepa_cifar10_v3.yaml \\
        --checkpoint checkpoints/baseline_v3/latest.pt

Default: cosine LR + sweep over {3e-4, 1e-3, 3e-3} with feature standardization.
Expected baseline: ~77.23% top-1, feat_std ~0.16 (~15–20 min on MPS).

Use ``--no-tuned`` for the cheaper fixed-LR probe (trend monitoring only).
"""
from __future__ import annotations

from jepa.cli.linear_probe import main

if __name__ == "__main__":
    main()
