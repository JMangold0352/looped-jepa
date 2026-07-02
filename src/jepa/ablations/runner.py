from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from jepa.ablations.registry import AblationSuite, AblationVariant
from jepa.data.cifar10 import build_dataloaders
from jepa.eval.linear_probe import load_encoder_from_checkpoint, run_linear_probe_tuned
from jepa.eval.loop_metrics import measure_loop_usage, training_stability_from_metrics
from jepa.masking import IJEPAMaskCollator
from jepa.train import train
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.seed import set_seed


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_variant_config(
    base_config_path: str,
    variant: AblationVariant,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Materialize a full training config for one ablation variant."""
    cfg = load_config(base_config_path)
    cfg["predictor"]["looped"] = True
    cfg["predictor"]["ouro"] = False
    cfg["predictor"].update(variant.predictor_overrides)
    cfg["train"].update(variant.train_overrides)
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    return cfg


def _mask_collator(cfg: dict[str, Any]) -> IJEPAMaskCollator:
    grid_size = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    mask_cfg = cfg["masking"]
    return IJEPAMaskCollator(
        grid_size=grid_size,
        num_target_blocks=mask_cfg.get("num_target_blocks", 4),
        target_scale=tuple(mask_cfg.get("target_scale", [0.15, 0.2])),
        context_scale=tuple(mask_cfg.get("context_scale", [0.85, 1.0])),
        fixed_context_patches=mask_cfg.get("fixed_context_patches", 32),
        fixed_target_patches=mask_cfg.get("fixed_target_patches", 16),
    )


def evaluate_variant(
    cfg: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run tuned probe + loop-usage stats for a trained checkpoint."""
    eval_cfg = cfg.get("eval", {}) or {}
    lr_grid = tuple(eval_cfg.get("tuned_lr_grid", (3e-4, 1e-3, 3e-3)))

    model = load_encoder_from_checkpoint(cfg, checkpoint, device)
    probe = run_linear_probe_tuned(
        cfg,
        checkpoint,
        device=device,
        epochs=eval_cfg.get("tuned_probe_epochs", 100),
        lr_grid=lr_grid,
    )

    _, val_loader = build_dataloaders(
        cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        train_augment=False,
    )
    loop_usage = measure_loop_usage(model, val_loader, _mask_collator(cfg), device)

    metrics_path = Path(cfg["train"]["run_dir"]) / "metrics.jsonl"
    stability = training_stability_from_metrics(str(metrics_path))

    return {
        "checkpoint": str(checkpoint),
        "top1_accuracy": probe["top1_accuracy"],
        "best_lr": probe["best_lr"],
        "feat_std": probe["feat_std"],
        "results_by_lr": probe["results_by_lr"],
        "loop_usage": loop_usage,
        "training_stability": stability,
    }


def _checkpoint_epoch(path: Path) -> int:
    if not path.exists():
        return 0
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt.get("epoch", 0))


def run_variant(
    base_config_path: str,
    variant: AblationVariant,
    device: torch.device,
    *,
    train_run: bool = True,
    skip_existing: bool = True,
    epochs: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Train (optional) and evaluate a single ablation variant."""
    cfg = build_variant_config(base_config_path, variant, epochs=epochs)
    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / "latest.pt"

    if variant.existing_checkpoint and Path(variant.existing_checkpoint).exists():
        ckpt_path = Path(variant.existing_checkpoint)
        train_run = False

    train_time_sec: float | None = None
    target_epochs = cfg["train"]["epochs"]

    if train_run:
        ckpt_epoch = _checkpoint_epoch(ckpt_path)
        training_complete = ckpt_path.exists() and ckpt_epoch >= target_epochs

        if skip_existing and training_complete:
            print(f"  [skip train] complete checkpoint: {ckpt_path} (epoch {ckpt_epoch})")
        else:
            set_seed(cfg.get("seed", 42))
            resume_from = str(ckpt_path) if resume and ckpt_path.exists() else None
            start = time.perf_counter()
            ckpt_path = train(cfg, device, resume_from=resume_from)
            train_time_sec = time.perf_counter() - start
    elif not ckpt_path.exists():
        if not train_run:
            print(f"  [skip eval] no checkpoint: {ckpt_path}")
            return {
                "name": variant.name,
                "description": variant.description,
                "status": "missing_checkpoint",
                "checkpoint": str(ckpt_path),
            }
        raise FileNotFoundError(f"No checkpoint for {variant.name}: {ckpt_path}")

    eval_result = evaluate_variant(cfg, ckpt_path, device)
    return {
        "name": variant.name,
        "description": variant.description,
        "config": {
            "predictor": cfg["predictor"],
            "train": {
                "epochs": cfg["train"]["epochs"],
                "exit_entropy_beta": cfg["train"].get("exit_entropy_beta", 0.0),
                "run_dir": cfg["train"]["run_dir"],
                "checkpoint_dir": cfg["train"]["checkpoint_dir"],
            },
        },
        "train_time_sec": train_time_sec,
        **eval_result,
    }


def run_suite(
    suite: AblationSuite,
    base_config_path: str,
    device: torch.device,
    *,
    train_run: bool = True,
    skip_existing: bool = True,
    epochs: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Train and evaluate every variant in a suite."""
    results = []
    for variant in suite.variants:
        print(f"\n=== {suite.key} / {variant.name} ===")
        entry = run_variant(
            base_config_path,
            variant,
            device,
            train_run=train_run,
            skip_existing=skip_existing,
            epochs=epochs,
            resume=resume,
        )
        results.append(entry)
        if entry.get("status") == "missing_checkpoint":
            print(f"  pending: no checkpoint yet")
        else:
            print(
                f"  top1={entry['top1_accuracy']:.2f}%  "
                f"feat_std={entry['feat_std']:.4f}  "
                f"mean_loops={entry['loop_usage']['mean_loops_used']:.2f}"
            )

    payload = {
        "suite": suite.key,
        "title": suite.title,
        "base_config": base_config_path,
        "variants": results,
    }
    return payload
