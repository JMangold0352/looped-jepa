#!/usr/bin/env python3
"""Tuned linear probe on a frozen JEPA encoder (official evaluation metric).

Usage::

    python scripts/linear_probe.py \\
        --config configs/image_jepa_cifar10_v3.yaml \\
        --checkpoint checkpoints/baseline_v3/latest.pt

Default: cosine LR + sweep over {3e-4, 1e-3, 3e-3} with feature standardization.
Expected baseline: ~77.23% top-1, feat_std ~0.16 (~15–20 min on MPS).

Use ``--no-tuned`` for the cheaper fixed-LR probe (trend monitoring only).
"""
from __future__ import annotations

import argparse
import json

from jepa.eval.linear_probe import run_linear_probe, run_linear_probe_tuned
from jepa.utils.config import load_config
from jepa.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe on frozen JEPA encoder")
    parser.add_argument("--config", type=str, default="configs/image_jepa_cifar10.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--tuned",
        action="store_true",
        default=True,
        help="Run tuned probe with LR sweep (default).",
    )
    parser.add_argument(
        "--no-tuned",
        dest="tuned",
        action="store_false",
        help="Use the simple fixed-LR probe instead of the tuned sweep.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs for the tuned probe (ignored if --no-tuned).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))

    if args.tuned:
        results = run_linear_probe_tuned(
            cfg, args.checkpoint, device=device, epochs=args.epochs
        )
        print(
            f"Tuned linear probe top-1 accuracy: {results['top1_accuracy']:.2f}% "
            f"(best_lr={results['best_lr']:.0e}, feat_std={results['feat_std']:.4f})"
        )
        print(f"Per-LR results: {json.dumps(results['results_by_lr'], indent=2)}")
    else:
        results = run_linear_probe(cfg, args.checkpoint, device=device)
        print(f"Simple linear probe top-1 accuracy: {results['top1_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
