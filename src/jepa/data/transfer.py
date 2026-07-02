from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from jepa.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD


class FolderClassificationDataset(Dataset):
    """Image-folder dataset: ``root/class_name/image.ext`` layout."""

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions
        self.samples: list[tuple[Path, int]] = []
        self.classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        for class_name in self.classes:
            class_dir = self.root / class_name
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in self.extensions:
                    self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transfer_transforms(
    img_size: int = 32,
    train: bool = True,
) -> transforms.Compose:
    """Resize + normalize transforms for downstream transfer datasets."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(img_size, padding=max(4, img_size // 8)),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_folder_dataloaders(
    data_dir: str | Path,
    batch_size: int = 128,
    img_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Build train/val loaders from ``train/`` and ``val/`` subfolders.

    If only ``train/`` exists, a stratified holdout split is created from it.
    Returns ``(train_loader, val_loader, metadata)`` where metadata includes
    ``num_classes`` and ``class_names``.
    """
    root = Path(data_dir)
    train_dir = root / "train"
    val_dir = root / "val"

    if train_dir.is_dir() and val_dir.is_dir():
        train_ds = FolderClassificationDataset(
            train_dir, transform=build_transfer_transforms(img_size, train=True)
        )
        val_ds = FolderClassificationDataset(
            val_dir, transform=build_transfer_transforms(img_size, train=False)
        )
        class_names = train_ds.classes
        num_classes = len(class_names)
    else:
        eval_ds = FolderClassificationDataset(
            root, transform=build_transfer_transforms(img_size, train=False)
        )
        train_only_ds = FolderClassificationDataset(
            root, transform=build_transfer_transforms(img_size, train=True)
        )
        if not train_only_ds.samples:
            raise FileNotFoundError(
                f"No images found under {root}. Expected class subfolders like {root}/cat/*.jpg"
            )
        num_classes = len(train_only_ds.classes)
        class_names = train_only_ds.classes
        generator = torch.Generator().manual_seed(seed)
        val_size = max(1, int(len(train_only_ds) * val_split))
        perm = torch.randperm(len(train_only_ds), generator=generator).tolist()
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]
        train_ds = torch.utils.data.Subset(train_only_ds, train_idx)
        val_ds = torch.utils.data.Subset(eval_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    metadata = {"num_classes": num_classes, "class_names": class_names, "img_size": img_size}
    return train_loader, val_loader, metadata


def build_cifar100_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    img_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """CIFAR-100 as a lightweight transfer benchmark (100 classes, same 32x32 scale)."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_tf = build_transfer_transforms(img_size, train=True)
    val_tf = build_transfer_transforms(img_size, train=False)
    train_ds = datasets.CIFAR100(root=str(root), train=True, download=True, transform=train_tf)
    val_ds = datasets.CIFAR100(root=str(root), train=False, download=True, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    metadata = {
        "num_classes": 100,
        "class_names": [str(i) for i in range(100)],
        "img_size": img_size,
        "dataset": "cifar100",
    }
    return train_loader, val_loader, metadata


def build_stl10_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    img_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """STL-10 as a real-world-ish transfer benchmark (10 classes, 96x96 downscaled)."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_tf = build_transfer_transforms(img_size, train=True)
    val_tf = build_transfer_transforms(img_size, train=False)
    train_ds = datasets.STL10(root=str(root), split="train", download=True, transform=train_tf)
    val_ds = datasets.STL10(root=str(root), split="test", download=True, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    metadata = {
        "num_classes": 10,
        "class_names": [str(i) for i in range(10)],
        "img_size": img_size,
        "dataset": "stl10",
    }
    return train_loader, val_loader, metadata
