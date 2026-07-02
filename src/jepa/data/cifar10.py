from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 channel statistics (computed over the train split).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _normalize() -> transforms.Normalize:
    return transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def build_transforms(
    train: bool = True,
    augmentation: dict[str, Any] | None = None,
) -> transforms.Compose:
    """Build CIFAR-10 transforms.

    ``train=False`` always returns the deterministic eval transform (tensor +
    normalize). ``train=True`` returns a training transform; the recipe is
    selected by the ``augmentation`` dict (see below). When ``augmentation`` is
    ``None`` the legacy v1 recipe (flip + crop + stochastic ColorJitter) is
    used so existing configs keep their behavior.

    ``augmentation`` schema (all keys optional):

    - ``kind``: ``"default"`` | ``"randaugment"`` | ``"aggressive_color_jitter"``.
      ``"default"`` matches v1. ``"randaugment"`` adds RandAugment + a
      conservative RandomResizedCrop. ``"aggressive_color_jitter"`` keeps the
      v1 flip+crop but widens ColorJitter and adds a RandomResizedCrop.
    - ``randaugment_n`` (int, default 2): number of RandAugment ops.
    - ``randaugment_m`` (int, default 9): magnitude in [0, 30].
    - ``rrc_scale`` ([lo, hi], default [0.5, 1.0]): RandomResizedCrop area
      scale range. Kept conservative for 32x32 so patch structure survives.
    - ``rrc_ratio`` ([lo, hi], default [0.8, 1.25]): aspect ratio range.
    - ``color_jitter`` ([b, c, s, h], default [0.4, 0.4, 0.4, 0.1]).
    - ``color_jitter_p`` (float, default 0.8): probability of applying jitter.
    - ``rrc_after_ra`` (bool, default False): if True, RandomResizedCrop runs
      *after* RandAugment; by default it runs first so augmentation acts on a
      stable crop.
    - ``random_erasing_p`` (float, default 0.0): if > 0, append
      RandomErasing with this probability after normalization. Applied to the
      tensor (post-normalize) per the standard recipe.
    - ``random_erasing_scale`` ([lo, hi], default [0.02, 0.33]): erasing area
      range as a fraction of the image area.
    - ``random_erasing_ratio`` ([lo, hi], default [0.3, 3.3]): erasing aspect
      ratio range.
    """
    if not train:
        return transforms.Compose([transforms.ToTensor(), _normalize()])

    aug = augmentation or {}
    kind = aug.get("kind", "default")

    # Optional RandomErasing tail (applied post-normalize, on the tensor).
    re_p = float(aug.get("random_erasing_p", 0.0))
    re_scale = tuple(aug.get("random_erasing_scale", [0.02, 0.33]))
    re_ratio = tuple(aug.get("random_erasing_ratio", [0.3, 3.3]))
    tail: list[transforms.Transform] = [transforms.ToTensor(), _normalize()]
    if re_p > 0.0:
        tail.append(
            transforms.RandomErasing(p=re_p, scale=re_scale, ratio=re_ratio)
        )

    if kind == "default":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                *tail,
            ]
        )

    # Tunable knobs with CIFAR-10-appropriate defaults.
    rrp_scale = tuple(aug.get("rrc_scale", [0.5, 1.0]))
    rrp_ratio = tuple(aug.get("rrc_ratio", [0.8, 1.25]))
    color_jitter = aug.get("color_jitter", [0.4, 0.4, 0.4, 0.1])
    color_jitter_p = float(aug.get("color_jitter_p", 0.8))

    # Build the optional RandomResizedCrop. At 32x32 keep the scale range
    # conservative and use bilinear interpolation; aggressive RRC (e.g. 0.08)
    # destroys the small-scale structure I-JEPA relies on for patch prediction.
    def _rrc() -> transforms.RandomResizedCrop:
        return transforms.RandomResizedCrop(
            32, scale=rrp_scale, ratio=rrp_ratio, antialias=True
        )

    if kind == "aggressive_color_jitter":
        # v1 geometry (flip + crop) + widened jitter + RRC. Cheaper than
        # RandAugment and a good baseline if RA is unavailable.
        steps: list[transforms.Transform] = [transforms.RandomHorizontalFlip()]
        if aug.get("use_rrc", True):
            steps.append(_rrc())
        else:
            steps.append(transforms.RandomCrop(32, padding=4))
        steps.append(
            transforms.RandomApply(
                [transforms.ColorJitter(*color_jitter)], p=color_jitter_p
            )
        )
        steps.extend(tail)
        return transforms.Compose(steps)

    if kind == "randaugment":
        randaug_n = int(aug.get("randaugment_n", 2))
        randaug_m = int(aug.get("randaugment_m", 9))
        steps = [transforms.RandomHorizontalFlip()]
        if not aug.get("rrc_after_ra", False) and aug.get("use_rrc", True):
            steps.append(_rrc())
        steps.append(transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m))
        if aug.get("rrc_after_ra", False) and aug.get("use_rrc", True):
            steps.append(_rrc())
        steps.extend(tail)
        return transforms.Compose(steps)

    raise ValueError(f"Unknown augmentation kind: {kind!r}")


def build_weak_transforms() -> transforms.Compose:
    """Weak augmentation for the teacher view in two-view I-JEPA.

    The teacher should see a stable, lightly-augmented version of the image so
    its patch embeddings are predictable targets. Only horizontal flip + a
    small padded crop (the classic CIFAR-10 weak aug). No color jitter, no
    RandAugment, no RRC, no erasing.
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            _normalize(),
        ]
    )


class TwoViewTransform:
    """Apply independent strong + weak transforms to one PIL image.

    Used by two-view I-JEPA (v4): the student context encoder consumes the
    strong view, the EMA target encoder consumes the weak view. Returning a
    ``(strong, weak)`` tuple lets the standard DataLoader collate stacks of
    pairs without a custom collate_fn.
    """

    def __init__(self, strong: transforms.Compose, weak: transforms.Compose) -> None:
        self.strong = strong
        self.weak = weak

    def __call__(self, image):
        return self.strong(image), self.weak(image)


def build_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    train_augment: bool = True,
    augmentation: dict[str, Any] | None = None,
    two_view: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build CIFAR-10 train/val dataloaders.

    ``train_augment=False`` (used by the linear probe) forces the deterministic
    eval transform on the train split too, so the probe measures representation
    quality rather than augmentation robustness. ``augmentation`` is forwarded
    to :func:`build_transforms` only when ``train_augment`` is True.

    ``two_view=True`` wraps the train transform in :class:`TwoViewTransform` so
    each train sample yields ``(strong_view, weak_view)`` tensors. The val
    loader is never affected. Probe loaders should always pass
    ``train_augment=False`` (and ``two_view=False``).
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    if two_view:
        if not train_augment:
            raise ValueError("two_view=True requires train_augment=True")
        train_tf = TwoViewTransform(
            strong=build_transforms(train=True, augmentation=augmentation),
            weak=build_weak_transforms(),
        )
    else:
        train_tf = build_transforms(
            train=train_augment,
            augmentation=augmentation if train_augment else None,
        )

    train_ds = datasets.CIFAR10(
        root=str(root), train=True, download=True, transform=train_tf
    )
    val_ds = datasets.CIFAR10(
        root=str(root), train=False, download=True, transform=build_transforms(False)
    )

    # With workers, keep them alive across epochs and prefetch a few batches so
    # CPU-side augmentation (ColorJitter / RandAugment especially) overlaps
    # with GPU compute.
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
