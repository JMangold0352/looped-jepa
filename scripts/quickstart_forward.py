#!/usr/bin/env python3
"""Quick sanity check: load a released model and print feat_std on 64 val images.

The probe metric in training uses train-split statistics; this script reports
mean per-dimension std on the first ``--n`` CIFAR-10 validation images only,
for a fast local check after install or download.

Usage::

    python scripts/quickstart_forward.py --model baseline_v3
    python scripts/quickstart_forward.py --model looped_v3 --device mps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader, Subset

from jepa import load_ijepa
from jepa.data.cifar10 import build_dataloaders
from jepa.eval.linear_probe import extract_features
from jepa.utils.device import get_device
from jepa.utils.weights import get_released_weight


def feat_std_on_features(feats) -> float:
    return feats.std(dim=0).mean().item()


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward pass + feat_std on CIFAR-10 val subset")
    parser.add_argument(
        "--model",
        default="baseline_v3",
        choices=["baseline_v3", "looped_v3", "sandwich_rmsnorm"],
        help="Released registry key",
    )
    parser.add_argument("--n", type=int, default=64, help="Number of validation images")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path (skips download)",
    )
    args = parser.parse_args()

    spec = get_released_weight(args.model)
    device = get_device(args.device)

    print(f"Loading {args.model} (reference top1={spec.tuned_top1:.2f}%, feat_std={spec.feat_std:.4f})")
    model = load_ijepa(
        args.model,
        pretrained=True,
        device=device,
        checkpoint=args.checkpoint,
    )

    _, val_loader = build_dataloaders(batch_size=min(args.n, 128), train_augment=False, num_workers=0)
    n = min(args.n, len(val_loader.dataset))
    subset = Subset(val_loader.dataset, list(range(n)))
    loader = DataLoader(subset, batch_size=n, shuffle=False, num_workers=0)

    feats, labels = extract_features(model, loader, device)
    feat_std = feat_std_on_features(feats)

    print(f"samples={len(labels)}  device={device}")
    print(f"feat_std={feat_std:.4f}  (registry reference on full tuned probe: {spec.feat_std:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
