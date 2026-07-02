"""Transfer-learning evaluation: metrics, scratch baseline, qualitative outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from jepa.eval.linear_probe import (
    LinearProbeHead,
    _standardize_features,
    _train_probe_head,
    extract_features,
    load_encoder_from_checkpoint,
    probe_model_tuned,
)


def _num_classes_from_loader(loader: DataLoader) -> int:
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return len(ds.classes)


def _class_names_from_loader(loader: DataLoader) -> list[str]:
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return list(ds.classes)


def _macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    f1s: list[float] = []
    for c in range(num_classes):
        tp = np.sum((preds_np == c) & (labels_np == c))
        fp = np.sum((preds_np == c) & (labels_np != c))
        fn = np.sum((preds_np != c) & (labels_np == c))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))


def _probe_macro_f1(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    embed_dim: int,
    num_classes: int,
    best_lr: float,
    epochs: int = 50,
) -> float:
    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)
    train_feats, val_feats, _ = _standardize_features(train_feats, val_feats)
    head, _acc = _train_probe_head(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        embed_dim=embed_dim,
        device=device,
        epochs=epochs,
        probe_lr=best_lr,
        weight_decay=1e-4,
        num_classes=num_classes,
        cosine_schedule=True,
    )
    head.eval()
    with torch.no_grad():
        preds = head(val_feats.to(device)).argmax(dim=1).cpu()
    return 100.0 * _macro_f1(preds, val_labels, num_classes)


def evaluate_probe_head(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    head.eval()
    with torch.no_grad():
        logits = head(features.to(device))
        preds = logits.argmax(dim=1).cpu()
    acc = 100.0 * (preds == labels).float().mean().item()
    return {
        "top1_accuracy": acc,
        "macro_f1": 100.0 * _macro_f1(preds, labels, num_classes) / 1.0,
    }


def run_frozen_encoder_transfer(
    cfg: dict[str, Any],
    checkpoint: str | Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    label: str,
    probe_epochs: int = 50,
) -> dict[str, Any]:
    """Tuned linear probe on a frozen JEPA encoder."""
    model = load_encoder_from_checkpoint(cfg, checkpoint, device)
    num_classes = _num_classes_from_loader(train_loader)

    results = probe_model_tuned(
        model,
        train_loader,
        val_loader,
        device,
        embed_dim=cfg["encoder"]["embed_dim"],
        epochs=probe_epochs,
        weight_decay=cfg["eval"].get("probe_weight_decay", 1e-4),
        num_classes=num_classes,
    )
    results["macro_f1"] = _probe_macro_f1(
        model,
        train_loader,
        val_loader,
        device,
        cfg["encoder"]["embed_dim"],
        num_classes,
        best_lr=results["best_lr"],
        epochs=min(probe_epochs, 50),
    )
    results["method"] = label
    results["checkpoint"] = str(checkpoint)
    return results


class ScratchResNet18(nn.Module):
    """Lightweight ResNet18 for small-image classification from scratch."""

    def __init__(self, num_classes: int, img_size: int = 96) -> None:
        super().__init__()
        self.img_size = img_size
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.net = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_scratch_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    img_size: int = 96,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    """Train a small ResNet18 from scratch on the transfer dataset."""
    model = ScratchResNet18(num_classes, img_size=img_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1).cpu()
                correct += (preds == labels).sum().item()
                total += labels.numel()
                all_preds.append(preds)
                all_labels.append(labels)
        acc = 100.0 * correct / max(1, total)
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [scratch] epoch {epoch + 1}/{epochs}  val_top1={acc:.2f}%")

    if best_state:
        model.load_state_dict(best_state)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    macro_f1 = 100.0 * _macro_f1(preds, labels, num_classes)
    return {
        "method": "scratch_resnet18",
        "top1_accuracy": best_acc,
        "macro_f1": macro_f1,
        "epochs": epochs,
    }


def encoder_gradcam(
    model,
    image: torch.Tensor,
    probe_head: LinearProbeHead,
    grid_size: int,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """Probe-weighted patch saliency (Grad-CAM-style, no encoder grad required)."""
    model.eval()
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = model.encoder.forward_all_patches(image)  # (1, N, D)
        weights = probe_head.head.weight[target_class]
        cam = (tokens[0] * weights).sum(dim=-1).clamp(min=0).detach().cpu().numpy()
    cam = cam.reshape(grid_size, grid_size)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def plot_qualitative_grid(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: list[str],
    output_path: Path,
    title: str,
    cams: list[np.ndarray] | None = None,
) -> None:
    """Save a grid of predictions (optional Grad-CAM overlays)."""
    n = min(8, images.shape[0])
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    for i in range(len(axes)):
        ax = axes[i]
        ax.set_axis_off()
        if i >= n:
            continue
        img = images[i].cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        if cams and i < len(cams):
            ax.imshow(cams[i], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        t = class_names[int(true_labels[i])]
        p = class_names[int(pred_labels[i])]
        color = "white" if t == p else "#ffcccc"
        ax.set_title(f"T:{t}\nP:{p}", fontsize=8, color=color)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_results_markdown(
    output_path: Path,
    dataset_meta: dict[str, Any],
    rows: list[dict[str, Any]],
    defense_paragraph: str,
    qualitative_note: str,
) -> None:
    lines = [
        "# Transfer Learning: Roboflow Aerial Maritime",
        "",
        f"**Dataset**: {dataset_meta.get('name', 'unknown')}",
        f"**Classes**: {dataset_meta.get('num_classes', '?')} ({', '.join(dataset_meta.get('class_names', []))})",
        f"**Train / val images**: {dataset_meta.get('train_size', '?')} / {dataset_meta.get('val_size', '?')}",
        "",
        "## Results",
        "",
        "| Method | Top-1 (%) | Macro F1 (%) | Notes |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['top1_accuracy']:.2f} | {row.get('macro_f1', 0):.2f} | {row.get('notes', '')} |"
        )
    lines.extend(
        [
            "",
            "## Relevance to Defense & Autonomy",
            "",
            defense_paragraph,
            "",
            "## Qualitative",
            "",
            qualitative_note,
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def save_results_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
