"""Dataset dispatch for training and probing."""
from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from jepa.data.cifar10 import build_dataloaders as build_cifar10_dataloaders


def build_dataloaders_from_config(
    cfg: dict[str, Any],
    *,
    train_augment: bool = True,
    two_view: bool = False,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders from ``cfg['data']`` (default dataset: cifar10)."""
    data_cfg = cfg["data"]
    dataset = data_cfg.get("dataset", "cifar10")
    bs = batch_size if batch_size is not None else data_cfg["batch_size"]
    nw = num_workers if num_workers is not None else data_cfg.get("num_workers", 0)

    if dataset == "cifar10":
        return build_cifar10_dataloaders(
            data_cfg["data_dir"],
            batch_size=bs,
            num_workers=nw,
            train_augment=train_augment,
            augmentation=cfg.get("augmentation") if train_augment else None,
            two_view=two_view,
        )

    if dataset == "tiny_imagenet":
        from jepa.data.tiny_imagenet import build_dataloaders as build_tiny_dataloaders

        return build_tiny_dataloaders(
            data_cfg["data_dir"],
            batch_size=bs,
            num_workers=nw,
            train_augment=train_augment,
            augmentation=cfg.get("augmentation") if train_augment else None,
            img_size=data_cfg.get("img_size", 64),
            two_view=two_view,
        )

    raise ValueError(f"Unknown dataset {dataset!r} in config data.dataset")


def num_classes_from_config(cfg: dict[str, Any]) -> int:
    return int(cfg["data"].get("num_classes", 10))
