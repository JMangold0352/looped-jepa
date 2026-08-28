#!/usr/bin/env python3
"""Tuned linear probe for scale-comparison datasets (CIFAR-10, Tiny ImageNet, etc.).

Uses the same protocol as ``scripts/linear_probe.py``: frozen encoder, feature
standardization, cosine LR schedule, sweep over {3e-4, 1e-3, 3e-3}.

Usage::

    python scripts/scale_probe.py \\
        --config configs/image_jepa_tinyimagenet_v3.yaml \\
        --checkpoint checkpoints/tinyimagenet_baseline_v3/latest.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa.eval.linear_probe import run_linear_probe_tuned
from jepa.utils.cli import require_file
from jepa.utils.config import load_config
from jepa.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Tuned linear probe for scale experiments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional JSON path to append results (e.g. results/scale/runs.json)",
    )
    args = parser.parse_args()

    require_file(args.config, label="Config file")
    require_file(args.checkpoint, label="Checkpoint file")

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    results = run_linear_probe_tuned(
        cfg, args.checkpoint, device=device, epochs=args.epochs
    )
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dataset": cfg["data"].get("dataset", "cifar10"),
        "num_classes": cfg["data"].get("num_classes", 10),
        **results,
    }
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if out_path.is_file():
            existing = json.loads(out_path.read_text())
        existing.append(payload)
        out_path.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
