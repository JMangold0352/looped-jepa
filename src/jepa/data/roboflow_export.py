"""Load Roboflow exports (folder layout or YOLO detection → image classification)."""
from __future__ import annotations

import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

import torch
import urllib.request
from torch.utils.data import DataLoader

from jepa.data.transfer import build_folder_dataloaders


def download_roboflow_universe(
    workspace: str,
    project: str,
    version: int,
    dest_dir: str | Path,
    model_format: str = "folder",
) -> Path:
    """Download a public Roboflow Universe dataset (requires ``ROBOFLOW_API_KEY``)."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set ROBOFLOW_API_KEY (free at https://app.roboflow.com/settings/api) "
            "or pass --export-url / --data-dir."
        )
    from roboflow import Roboflow

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download(
        model_format=model_format,
        location=str(dest),
    )
    return Path(dataset.location)


def download_zip_export(url: str, dest_dir: str | Path) -> Path:
    """Download and extract a Roboflow zip export URL."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "roboflow_export.zip"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink(missing_ok=True)
    return dest


def _parse_yolo_label(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    classes: list[int] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            classes.append(int(parts[0]))
    return classes


def _dominant_class(class_ids: list[int]) -> int | None:
    if not class_ids:
        return None
    counts: dict[int, int] = {}
    for cid in class_ids:
        counts[cid] = counts.get(cid, 0) + 1
    return max(counts, key=counts.get)


def yolo_detection_to_classification(
    yolo_root: str | Path,
    out_dir: str | Path,
    min_objects: int = 1,
) -> dict[str, Any]:
    """Convert a YOLO detection export to folder classification (dominant class / image)."""
    root = Path(yolo_root)
    data_yaml = root / "data.yaml"
    class_names: list[str] = []
    if data_yaml.exists():
        for line in data_yaml.read_text().splitlines():
            if line.strip().startswith("names:"):
                # names: ['boat', 'dock', ...]
                names_part = line.split(":", 1)[1].strip()
                import ast

                class_names = ast.literal_eval(names_part)
                break

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    stats = {"splits": {}, "class_names": class_names}
    for split in ("train", "valid", "val", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir():
            continue
        if split in ("valid", "val"):
            out_split = "val"
        elif split == "test" and not (out / "val").exists():
            out_split = "val"
        else:
            out_split = "train" if split == "train" else split
        split_out = out / out_split
        split_out.mkdir(parents=True, exist_ok=True)
        kept = 0
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            label_path = lbl_dir / f"{img_path.stem}.txt"
            class_ids = _parse_yolo_label(label_path)
            if len(class_ids) < min_objects:
                continue
            dom = _dominant_class(class_ids)
            if dom is None:
                continue
            if class_names and dom < len(class_names):
                class_name = class_names[dom]
            else:
                class_name = f"class_{dom}"
            class_dir = split_out / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, class_dir / img_path.name)
            kept += 1
        stats["splits"][out_split] = kept

    if not class_names:
        # Infer from folders
        train_root = out / "train"
        if train_root.is_dir():
            class_names = sorted(d.name for d in train_root.iterdir() if d.is_dir())
            stats["class_names"] = class_names

    return stats


def resolve_classification_dir(raw_dir: Path) -> Path:
    """Return a folder-classification root (train/<class>/...) from various Roboflow layouts."""
    raw_dir = Path(raw_dir)
    if (raw_dir / "train").is_dir() and any((raw_dir / "train").iterdir()):
        first = next((raw_dir / "train").iterdir())
        if first.is_dir():
            return raw_dir

    # YOLO layout
    for split in ("train", "valid"):
        if (raw_dir / split / "images").is_dir():
            cls_dir = raw_dir / "classification"
            yolo_detection_to_classification(raw_dir, cls_dir)
            return cls_dir

    # Nested single project folder from roboflow download
    for child in raw_dir.iterdir():
        if child.is_dir() and (child / "train").is_dir():
            return resolve_classification_dir(child)

    raise FileNotFoundError(
        f"No usable classification layout under {raw_dir}. "
        "Expected train/<class>/ or YOLO train/images + labels."
    )


def build_eurosat_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    img_size: int = 32,
    num_workers: int = 0,
    max_train: int | None = 1500,
    max_val: int | None = 400,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """EuroSAT RGB: satellite land-cover (small download, aerial ISR proxy)."""
    from torchvision.datasets import EuroSAT

    from jepa.data.transfer import build_transfer_transforms

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_tf = build_transfer_transforms(img_size, train=True)
    val_tf = build_transfer_transforms(img_size, train=False)
    full_train = EuroSAT(root=str(root), download=True, transform=train_tf)
    full_val = EuroSAT(root=str(root), download=True, transform=val_tf)

    if max_train and len(full_train) > max_train:
        idx = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(42))[:max_train].tolist()
        train_ds = torch.utils.data.Subset(full_train, idx)
    else:
        train_ds = full_train
    if max_val and len(full_val) > max_val:
        idx = torch.randperm(len(full_val), generator=torch.Generator().manual_seed(7))[:max_val].tolist()
        val_ds = torch.utils.data.Subset(full_val, idx)
    else:
        val_ds = full_val

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    classes = full_train.classes
    meta = {
        "num_classes": len(classes),
        "class_names": list(classes),
        "img_size": img_size,
        "dataset": "eurosat",
        "name": "EuroSAT RGB (satellite aerial proxy; use Roboflow maritime with --download when API key set)",
    }
    return train_loader, val_loader, meta


def build_fgvc_aircraft_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    img_size: int = 32,
    num_workers: int = 0,
    max_train: int | None = 2000,
    max_val: int | None = 500,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """FGVC Aircraft: aerial vehicle classes (downloadable without Roboflow API)."""
    from torchvision.datasets import FGVCAircraft

    from jepa.data.transfer import build_transfer_transforms

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_tf = build_transfer_transforms(img_size, train=True)
    val_tf = build_transfer_transforms(img_size, train=False)
    train_ds = FGVCAircraft(root=str(root), split="trainval", annotation_level="variant", download=True, transform=train_tf)
    val_ds = FGVCAircraft(root=str(root), split="test", annotation_level="variant", download=True, transform=val_tf)

    if max_train and len(train_ds) > max_train:
        idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(42))[:max_train].tolist()
        train_ds = torch.utils.data.Subset(train_ds, idx)
    if max_val and len(val_ds) > max_val:
        idx = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(42))[:max_val].tolist()
        val_ds = torch.utils.data.Subset(val_ds, idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    classes = getattr(train_ds, "classes", None) or getattr(val_ds, "classes", None)
    if classes is None:
        base = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
        classes = getattr(base, "classes", [str(i) for i in range(100)])
    meta = {
        "num_classes": len(classes),
        "class_names": list(classes),
        "img_size": img_size,
        "dataset": "fgvc-aircraft",
        "name": "FGVC Aircraft (aerial vehicle proxy)",
    }
    return train_loader, val_loader, meta


def build_roboflow_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    img_size: int = 96,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Build loaders from a Roboflow folder or YOLO export directory."""
    cls_root = resolve_classification_dir(Path(data_dir))
    return build_folder_dataloaders(
        cls_root,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
    )
