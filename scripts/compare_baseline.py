#!/usr/bin/env python3
"""Compare linear-probe accuracy between two checkpoints (or vs random init)."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from jepa.eval.linear_probe import run_linear_probe, run_linear_probe_tuned
from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.seed import set_seed


def _probe(cfg, checkpoint: Path, device, tuned: bool) -> dict:
    if tuned:
        return run_linear_probe_tuned(cfg, checkpoint, device=device)
    return run_linear_probe(cfg, checkpoint, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/image_jepa_cifar10_v3.yaml")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline checkpoint (trained encoder)",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Candidate checkpoint to compare against baseline",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Single checkpoint (legacy alias for --baseline)",
    )
    parser.add_argument("--random", action="store_true", help="Also probe a random init")
    parser.add_argument("--tuned", action="store_true", help="Use tuned LR sweep probe")
    args = parser.parse_args()

    baseline_ckpt = args.baseline or args.checkpoint
    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 42))

    if baseline_ckpt:
        base = _probe(cfg, Path(baseline_ckpt), device, args.tuned)
        print(f"Baseline top-1: {base['top1_accuracy']:.2f}%  feat_std={base['feat_std']:.4f}")

    if args.candidate:
        cand = _probe(cfg, Path(args.candidate), device, args.tuned)
        print(f"Candidate top-1: {cand['top1_accuracy']:.2f}%  feat_std={cand['feat_std']:.4f}")
        if baseline_ckpt:
            delta = cand["top1_accuracy"] - base["top1_accuracy"]
            print(f"Delta: {delta:+.2f} pp")

    if args.random:
        with tempfile.TemporaryDirectory() as tmp:
            random_ckpt = Path(tmp) / "random_baseline.pt"
            model = IJEPA.from_config(cfg)
            torch.save({"model": model.state_dict()}, random_ckpt)
            rnd = _probe(cfg, random_ckpt, device, args.tuned)
        print(f"Random init top-1: {rnd['top1_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
