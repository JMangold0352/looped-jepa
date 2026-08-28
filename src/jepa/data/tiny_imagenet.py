"""Tiny ImageNet-200 loader (64×64 RGB, 200 classes).

Downloads ``tiny-imagenet-200.zip`` from the CS231n course page on first use.
Layout after extract::

    data/tiny-imagenet-200/
      train/<wnid>/images/*.JPEG
      val/images/*.JPEG
      val/val_annotations.txt
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
NUM_CLASSES = 200


def _normalize() -> transforms.Normalize:
    return transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def build_transforms(
    train: bool,
    img_size: int = 64,
    augmentation: dict[str, Any] | None = None,
) -> transforms.Compose:
    """v3-style aug adapted to ``img_size`` (default 64)."""
    if not train:
        return transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                _normalize(),
            ]
        )

    aug = augmentation or {}
    kind = aug.get("kind", "randaugment")
    rrc_scale = tuple(aug.get("rrc_scale", [0.5, 1.0]))
    rrc_ratio = tuple(aug.get("rrc_ratio", [0.8, 1.25]))

    def _rrc() -> transforms.RandomResizedCrop:
        return transforms.RandomResizedCrop(
            img_size, scale=rrc_scale, ratio=rrc_ratio, antialias=True
        )

    steps: list[transforms.Transform] = [transforms.RandomHorizontalFlip()]
    if kind == "randaugment":
        if not aug.get("rrc_after_ra", False) and aug.get("use_rrc", True):
            steps.append(_rrc())
        steps.append(
            transforms.RandAugment(
                num_ops=int(aug.get("randaugment_n", 2)),
                magnitude=int(aug.get("randaugment_m", 9)),
            )
        )
        if aug.get("rrc_after_ra", False) and aug.get("use_rrc", True):
            steps.append(_rrc())
    else:
        steps.append(transforms.RandomCrop(img_size, padding=max(4, img_size // 16)))

    steps.extend([transforms.ToTensor(), _normalize()])
    return transforms.Compose(steps)


def build_weak_transforms(img_size: int = 64) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=max(4, img_size // 16)),
            transforms.ToTensor(),
            _normalize(),
        ]
    )


class TwoViewTransform:
    def __init__(self, strong: transforms.Compose, weak: transforms.Compose) -> None:
        self.strong = strong
        self.weak = weak

    def __call__(self, image):
        return self.strong(image), self.weak(image)


class TinyImageNetValDataset(Dataset):
    """Validation split with labels from ``val_annotations.txt``."""

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform
        ann_path = root / "val" / "val_annotations.txt"
        wnid_to_idx = {
            name: i
            for i, name in enumerate(
                sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
            )
        }
        self.samples: list[tuple[Path, int]] = []
        with ann_path.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                fname, wnid = parts[0], parts[1]
                self.samples.append((root / "val" / "images" / fname, wnid_to_idx[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


class TinyImageNetTrainDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform
        classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.samples: list[tuple[Path, int]] = []
        for wnid in classes:
            img_dir = root / "train" / wnid / "images"
            for path in sorted(img_dir.glob("*.JPEG")):
                self.samples.append((path, self.class_to_idx[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


def ensure_tiny_imagenet(data_dir: str | Path) -> Path:
    """Download and extract Tiny ImageNet if missing."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "tiny-imagenet-200"
    if (extracted / "train").is_dir() and (extracted / "val" / "val_annotations.txt").is_file():
        return extracted

    zip_path = root / "tiny-imagenet-200.zip"
    if not zip_path.is_file():
        print(f"Downloading Tiny ImageNet to {zip_path} ...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return extracted


def build_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    train_augment: bool = True,
    augmentation: dict[str, Any] | None = None,
    img_size: int = 64,
    two_view: bool = False,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    root = Path(data_dir)
    if download:
        tiny_root = ensure_tiny_imagenet(root)
    else:
        tiny_root = root / "tiny-imagenet-200"
        if not tiny_root.is_dir():
            raise FileNotFoundError(
                f"Tiny ImageNet not found at {tiny_root}. "
                "Run with download=True or place the extracted folder there."
            )

    if two_view:
        if not train_augment:
            raise ValueError("two_view=True requires train_augment=True")
        train_tf = TwoViewTransform(
            strong=build_transforms(True, img_size, augmentation),
            weak=build_weak_transforms(img_size),
        )
    else:
        train_tf = build_transforms(train_augment, img_size, augmentation if train_augment else None)

    train_ds = TinyImageNetTrainDataset(tiny_root, train_tf)
    val_ds = TinyImageNetValDataset(
        tiny_root,
        build_transforms(False, img_size),
    )

    extra = (
        {"persistent_workers": True, "prefetch_factor": 4} if num_workers > 0 else {}
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        **extra,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        **extra,
    )
    return train_loader, val_loader
