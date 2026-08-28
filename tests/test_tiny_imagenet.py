"""Tiny ImageNet config smoke tests (no download)."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from jepa.data.tiny_imagenet import build_dataloaders
from jepa.masking import IJEPAMaskCollator
from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config


def _seed_tiny_imagenet(root: Path, n_classes: int = 2, n_train: int = 4) -> None:
    """Minimal Tiny ImageNet layout for loader tests."""
    tiny = root / "tiny-imagenet-200"
    for split in ("train", "val/images"):
        (tiny / split).mkdir(parents=True, exist_ok=True)

    wnids = [f"n{i:08d}" for i in range(n_classes)]
    for wnid in wnids:
        img_dir = tiny / "train" / wnid / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for j in range(n_train):
            path = img_dir / f"{wnid}_{j}.JPEG"
            Image.new("RGB", (64, 64), color=(j * 40 % 255, 50, 80)).save(path)

    ann_lines = []
    for i, wnid in enumerate(wnids):
        val_name = f"val_{i}.JPEG"
        Image.new("RGB", (64, 64), color=(100, i * 60, 120)).save(tiny / "val" / "images" / val_name)
        ann_lines.append(f"{val_name}\t{wnid}\n")
    (tiny / "val" / "val_annotations.txt").write_text("".join(ann_lines))


def test_tiny_imagenet_dataloader_builds(tmp_path: Path) -> None:
    _seed_tiny_imagenet(tmp_path)
    train_loader, val_loader = build_dataloaders(
        tmp_path,
        batch_size=2,
        num_workers=0,
        train_augment=False,
        download=False,
    )
    x, y = next(iter(val_loader))
    assert x.shape == (2, 3, 64, 64)
    assert y.ndim == 1


def test_tiny_imagenet_v3_config_forward() -> None:
    cfg = load_config("configs/image_jepa_tinyimagenet_v3.yaml")
    model = IJEPA.from_config(cfg)
    model.eval()
    params = model.num_trainable_params()
    assert 9_800_000 <= params <= 9_900_000  # patch-8 embed adds ~55k vs CIFAR patch-4

    b, img_size, patch = 2, cfg["data"]["img_size"], cfg["data"]["patch_size"]
    images = torch.randn(b, 3, img_size, img_size)
    grid = img_size // patch
    collator = IJEPAMaskCollator(
        grid_size=grid,
        fixed_context_patches=cfg["masking"]["fixed_context_patches"],
        fixed_target_patches=cfg["masking"]["fixed_target_patches"],
    )
    masks = collator(b)
    ctx = torch.stack(masks.context_indices)
    tgt = torch.stack(masks.target_indices)
    out = model(images, ctx, tgt)
    assert out["loss"].ndim == 0


def test_tiny_imagenet_looped_config_builds() -> None:
    cfg = load_config("configs/image_jepa_tinyimagenet_v3_looped.yaml")
    model = IJEPA.from_config(cfg)
    assert cfg["predictor"]["looped"] is True
    params = model.num_trainable_params()
    assert 9_800_000 <= params <= 9_900_000  # patch-8 embed adds ~55k vs CIFAR patch-4


def test_scale_table_json_has_cifar_and_tbd_rows() -> None:
    import json

    path = Path("results/scale/scale_table.json")
    data = json.loads(path.read_text())
    rows = {r["dataset"]: r for r in data["rows"]}
    assert "CIFAR-10" in rows
    assert rows["TinyImageNet-200"]["status"] == "TBD"
    assert "hypothesis" in data
