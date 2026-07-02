#!/usr/bin/env python3
"""Train I-JEPA on CIFAR-10 from a YAML config.

Usage::

    python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3.yaml
    python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3_looped.yaml

Auto-resumes from ``checkpoints/<checkpoint_dir>/latest.pt`` when present.
Use ``--no-auto-resume`` for a fresh run.

Outputs:
    - Checkpoint: ``train.checkpoint_dir`` / ``latest.pt``
    - Metrics: ``train.run_dir`` / ``metrics.jsonl``
    - Tuned linear probe at end of training (official metric)

Reproduces headline results when trained 300 epochs with the v3 recipe (~5–6 h MPS).
See REPRODUCTION.md and scripts/README.md.
"""

import argparse
from pathlib import Path

from jepa.train import train
from jepa.utils.config import load_config
from jepa.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train I-JEPA on CIFAR-10")
    parser.add_argument("--config", type=str, default="configs/image_jepa_cifar10.yaml")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (default: checkpoints/<run>/latest.pt if present)",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Train from scratch even if latest.pt exists in checkpoint_dir",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    resume_from = args.resume
    if resume_from is None and not args.no_auto_resume:
        latest = Path(cfg.get("train", {}).get("checkpoint_dir", "checkpoints")) / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    ckpt = train(cfg, device, resume_from=resume_from)
    print(f"Training complete. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
