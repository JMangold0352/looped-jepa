#!/usr/bin/env python3
"""Generate I-JEPA visualizations: metrics, masks, embeddings, probe sweeps.

Legacy single-checkpoint plots. For the full baseline-vs-looped publication suite,
use ``visualizations/generate_all_figures.py`` instead.

Usage::

    python scripts/visualize.py \\
        --config configs/image_jepa_cifar10_v3.yaml \\
        --checkpoint checkpoints/baseline_v3/latest.pt \\
        --out-dir runs/visualizations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jepa.data.cifar10 import build_dataloaders
from jepa.eval.linear_probe import extract_features, load_encoder_from_checkpoint, run_linear_probe_tuned
from jepa.masking import IJEPAMaskCollator
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.viz.plots import (
    plot_embedding_pca,
    plot_mask_overlay,
    plot_metrics,
    plot_probe_lr_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JEPA visualizations")
    parser.add_argument("--config", default="configs/image_jepa_cifar10_v3.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/baseline_v3/latest.pt")
    parser.add_argument("--metrics", default="runs/cifar10_baseline_v3/metrics.jsonl")
    parser.add_argument("--out-dir", default="runs/visualizations")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))

    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        plot_metrics(metrics_path, out_dir / "training_curves.png")
        print(f"Wrote {out_dir / 'training_curves.png'}")

    model = load_encoder_from_checkpoint(cfg, args.checkpoint, device)

    grid_size = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = IJEPAMaskCollator(
        grid_size=grid_size,
        fixed_context_patches=cfg["masking"].get("fixed_context_patches", 32),
        fixed_target_patches=cfg["masking"].get("fixed_target_patches", 16),
    )
    _, val_loader = build_dataloaders(
        cfg["data"]["data_dir"],
        batch_size=8,
        num_workers=0,
        train_augment=False,
    )
    images, _ = next(iter(val_loader))
    masks = collator(images.shape[0])
    plot_mask_overlay(
        images[0],
        masks.context_indices[0],
        masks.target_indices[0],
        grid_size,
        out_dir / "mask_overlay.png",
    )
    print(f"Wrote {out_dir / 'mask_overlay.png'}")

    feats, labels = extract_features(model, val_loader, device)
    plot_embedding_pca(feats, labels, out_dir / "embedding_pca.png")
    print(f"Wrote {out_dir / 'embedding_pca.png'}")

    tuned = run_linear_probe_tuned(cfg, args.checkpoint, device=device, epochs=50)
    plot_probe_lr_sweep(tuned["results_by_lr"], out_dir / "probe_lr_sweep.png")
    summary = {
        "top1_accuracy": tuned["top1_accuracy"],
        "best_lr": tuned["best_lr"],
        "feat_std": tuned["feat_std"],
        "results_by_lr": tuned["results_by_lr"],
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'probe_lr_sweep.png'}")
    print(f"Wrote {out_dir / 'probe_summary.json'}")


if __name__ == "__main__":
    main()
