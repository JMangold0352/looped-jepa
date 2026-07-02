#!/usr/bin/env python3
"""Offline tuned-probe runner that forces num_workers=0 so it works in sandboxes
that block the torch shared-memory manager (the default num_workers>0 path)."""
from __future__ import annotations

import argparse
import json

import torch

from jepa.eval.linear_probe import extract_features, probe_model_tuned
from jepa.data.cifar10 import build_dataloaders
from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/image_jepa_cifar10.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Force single-process data loading: the sandbox blocks torch_shm_manager.
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 256
    device = get_device(cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 42))
    print(f"device={device}  num_workers=0  batch_size={cfg['data']['batch_size']}")

    model = IJEPA.from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # strict=False: v1 checkpoint has predictor.tgt_proj.* which was removed in v2.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        print(f"  [probe] ignoring unexpected checkpoint keys: {unexpected}")
    if missing:
        print(f"  [probe] missing checkpoint keys (using init): {missing}")
    for p in model.encoder.parameters():
        p.requires_grad = False

    train_loader, val_loader = build_dataloaders(
        cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=0,
        train_augment=False,
    )

    results = probe_model_tuned(
        model,
        train_loader,
        val_loader,
        device,
        embed_dim=cfg["encoder"]["embed_dim"],
        epochs=args.epochs,
        lr_grid=(3e-4, 1e-3, 3e-3),
        weight_decay=cfg["eval"].get("probe_weight_decay", 1e-4),
    )
    print(
        f"\nTuned linear probe top-1 accuracy: {results['top1_accuracy']:.2f}% "
        f"(best_lr={results['best_lr']:.0e}, feat_std={results['feat_std']:.4f})"
    )
    print(f"Per-LR results: {json.dumps(results['results_by_lr'], indent=2)}")


if __name__ == "__main__":
    main()
