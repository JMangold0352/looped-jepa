#!/usr/bin/env python3
"""Compare tuned linear-probe accuracy: v3 baseline vs looped variant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa.eval.linear_probe import run_linear_probe_tuned
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.seed import set_seed


def evaluate_checkpoint(config: str, checkpoint: str, device, label: str) -> dict:
    cfg = load_config(config)
    results = run_linear_probe_tuned(cfg, checkpoint, device=device)
    return {
        "label": label,
        "config": config,
        "checkpoint": checkpoint,
        "top1_accuracy": results["top1_accuracy"],
        "best_lr": results["best_lr"],
        "feat_std": results["feat_std"],
        "results_by_lr": results["results_by_lr"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v3 baseline vs looped predictor")
    parser.add_argument("--baseline-config", default="configs/image_jepa_cifar10_v3.yaml")
    parser.add_argument("--baseline-checkpoint", default="checkpoints/baseline_v3/latest.pt")
    parser.add_argument("--looped-config", default="configs/image_jepa_cifar10_v3_looped.yaml")
    parser.add_argument("--looped-checkpoint", default="checkpoints/baseline_v3_looped/latest.pt")
    parser.add_argument("--out", default="runs/looped_v3_comparison.json")
    args = parser.parse_args()

    device = get_device("auto")
    set_seed(42)

    baseline = evaluate_checkpoint(
        args.baseline_config, args.baseline_checkpoint, device, "v3_baseline"
    )
    looped = evaluate_checkpoint(
        args.looped_config, args.looped_checkpoint, device, "v3_looped"
    )

    delta = looped["top1_accuracy"] - baseline["top1_accuracy"]
    payload = {
        "baseline": baseline,
        "looped": looped,
        "delta_top1": round(delta, 2),
        "summary": (
            f"v3 baseline {baseline['top1_accuracy']:.2f}% vs "
            f"looped {looped['top1_accuracy']:.2f}% "
            f"(delta {delta:+.2f} pp)"
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(payload["summary"])
    print(f"  baseline feat_std={baseline['feat_std']:.4f}")
    print(f"  looped   feat_std={looped['feat_std']:.4f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
