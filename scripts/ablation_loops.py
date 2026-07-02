#!/usr/bin/env python3
"""Sweep loop depth for the looped predictor and log probe accuracy."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from jepa.data.cifar10 import build_dataloaders
from jepa.eval.linear_probe import load_encoder_from_checkpoint, run_linear_probe
from jepa.masking import IJEPAMaskCollator
from jepa.models.jepa import IJEPA
from jepa.models.looped_predictor import LoopedPredictor
from jepa.train import stack_indices, train
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.seed import set_seed


@torch.no_grad()
def measure_latent_loss(model: IJEPA, loader, mask_collator, device, max_batches: int = 20) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        masks = mask_collator(images.shape[0])
        ctx = stack_indices(masks.context_indices, device)
        tgt = stack_indices(masks.target_indices, device)
        out = model(images, ctx, tgt)
        total += out["loss"].item()
        count += 1
    return total / max(1, count)


def run_ablation(config_path: str, loops: list[int], train_epochs: int = 2) -> None:
    base_cfg = load_config(config_path)
    device = get_device(base_cfg.get("device", "auto"))
    results = []

    for n_loops in loops:
        cfg = copy.deepcopy(base_cfg)
        cfg["predictor"]["looped"] = True
        cfg["predictor"]["ouro"] = True
        cfg["predictor"]["max_loops"] = n_loops
        cfg["predictor"]["use_exit_gate"] = n_loops > 1
        cfg["train"]["epochs"] = train_epochs
        cfg["train"]["run_dir"] = f"runs/ablation_loops_{n_loops}"
        cfg["train"]["checkpoint_dir"] = f"checkpoints/ablation_loops_{n_loops}"

        set_seed(cfg.get("seed", 42))
        start = time.perf_counter()
        ckpt_path = train(cfg, device)
        train_time = time.perf_counter() - start

        probe = run_linear_probe(cfg, ckpt_path, device=device, probe_epochs=5)

        _, val_loader = build_dataloaders(cfg["data"]["data_dir"], cfg["data"]["batch_size"])
        grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
        mask_collator = IJEPAMaskCollator(grid_size=grid, fixed_context_patches=32, fixed_target_patches=16)
        model = load_encoder_from_checkpoint(cfg, ckpt_path, device)
        if isinstance(model.predictor, LoopedPredictor):
            model.predictor.max_loops = n_loops
        latent_loss = measure_latent_loss(model, val_loader, mask_collator, device)

        entry = {
            "loops": n_loops,
            "latent_loss": latent_loss,
            "linear_probe_top1": probe["top1_accuracy"],
            "train_time_sec": train_time,
            "checkpoint": str(ckpt_path),
        }
        results.append(entry)
        print(json.dumps(entry, indent=2))

    out_path = Path("runs/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved ablation results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/image_jepa_ouro_looped.yaml")
    parser.add_argument("--loops", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    run_ablation(args.config, args.loops, train_epochs=args.epochs)


if __name__ == "__main__":
    main()
