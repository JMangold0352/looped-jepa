#!/usr/bin/env python3
"""Linear-probe a frozen JEPA encoder on a transfer dataset (folder or STL-10)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa.data.transfer import build_cifar100_dataloaders, build_folder_dataloaders, build_stl10_dataloaders
from jepa.eval.linear_probe import load_encoder_from_checkpoint, probe_model
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer linear probe on frozen encoder")
    parser.add_argument("--config", default="configs/image_jepa_cifar10_v3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        choices=("folder", "cifar100", "stl10"),
        default="cifar100",
        help="folder: class subfolders; cifar100: lightweight 100-class transfer; stl10: large download",
    )
    parser.add_argument("--data-dir", default="data/transfer")
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--out", default="runs/transfer_probe.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 42))

    if args.dataset == "cifar100":
        train_loader, val_loader, meta = build_cifar100_dataloaders(
            data_dir=cfg["data"]["data_dir"],
            batch_size=cfg["data"]["batch_size"],
            img_size=cfg["data"]["img_size"],
            num_workers=cfg["data"].get("num_workers", 0),
        )
    elif args.dataset == "stl10":
        train_loader, val_loader, meta = build_stl10_dataloaders(
            data_dir=cfg["data"]["data_dir"],
            batch_size=cfg["data"]["batch_size"],
            img_size=cfg["data"]["img_size"],
            num_workers=cfg["data"].get("num_workers", 0),
        )
    else:
        train_loader, val_loader, meta = build_folder_dataloaders(
            args.data_dir,
            batch_size=cfg["data"]["batch_size"],
            img_size=cfg["data"]["img_size"],
            num_workers=cfg["data"].get("num_workers", 0),
        )

    model = load_encoder_from_checkpoint(cfg, args.checkpoint, device)

    results = probe_model(
        model,
        train_loader,
        val_loader,
        device,
        embed_dim=cfg["encoder"]["embed_dim"],
        epochs=args.probe_epochs,
        num_classes=meta["num_classes"],
    )

    payload = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "num_classes": meta["num_classes"],
        "class_names": meta.get("class_names"),
        **results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"Transfer probe top-1: {results['top1_accuracy']:.2f}% "
        f"(feat_std={results['feat_std']:.4f})"
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
