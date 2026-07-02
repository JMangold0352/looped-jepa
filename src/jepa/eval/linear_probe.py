from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from jepa.data.cifar10 import build_dataloaders
from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.seed import set_seed


class LinearProbeHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


@torch.no_grad()
def extract_features(
    model: IJEPA, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool patch tokens from the frozen encoder for each batch in ``loader``."""
    was_training = model.training
    model.eval()
    feats, labels = [], []
    for images, y in loader:
        images = images.to(device)
        tokens = model.encoder.forward_all_patches(images)
        feats.append(tokens.mean(dim=1).cpu())
        labels.append(y)
    if was_training:
        model.train()
    return torch.cat(feats), torch.cat(labels)


def _standardize_features(
    train_feats: torch.Tensor, val_feats: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float]:
    mean = train_feats.mean(dim=0, keepdim=True)
    std = train_feats.std(dim=0, keepdim=True)
    feat_std = std.mean().item()
    std = std.clamp_min(1e-6)
    return (train_feats - mean) / std, (val_feats - mean) / std, feat_std


def _cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    warmup = max(1, int(0.05 * total_steps))
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def _train_probe_head(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    embed_dim: int,
    device: torch.device,
    epochs: int,
    probe_lr: float,
    weight_decay: float,
    num_classes: int,
    cosine_schedule: bool = False,
    seed: int = 0,
) -> tuple[LinearProbeHead, float]:
    """Train a linear head on pre-extracted features. Returns (head, val top-1 %)."""
    if cosine_schedule:
        torch.manual_seed(seed)

    feat_loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=512,
        shuffle=True,
    )
    head = LinearProbeHead(embed_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=probe_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_steps = epochs * len(feat_loader) if cosine_schedule else 0
    step = 0
    best_acc = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    val_feats_dev = val_feats.to(device)

    head.train()
    for _ in range(epochs):
        for feats, labels in feat_loader:
            feats, labels = feats.to(device), labels.to(device)
            if cosine_schedule:
                for g in optimizer.param_groups:
                    g["lr"] = _cosine_lr(step, total_steps, probe_lr)
            logits = head(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cosine_schedule:
                step += 1

        head.eval()
        with torch.no_grad():
            preds = head(val_feats_dev).argmax(dim=1).cpu()
            acc = 100.0 * (preds == val_labels).sum().item() / val_labels.numel()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        head.train()

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, best_acc if cosine_schedule else _eval_head(head, val_feats, val_labels, device)


def _eval_head(
    head: LinearProbeHead,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
) -> float:
    head.eval()
    with torch.no_grad():
        preds = head(val_feats.to(device)).argmax(dim=1).cpu()
    return 100.0 * (preds == val_labels).sum().item() / val_labels.numel()


def train_probe_head(
    model: IJEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    embed_dim: int,
    epochs: int = 20,
    probe_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_classes: int = 10,
) -> LinearProbeHead:
    """Fit a linear classifier on frozen encoder features and return the head."""
    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)
    train_feats, val_feats, _ = _standardize_features(train_feats, val_feats)
    head, _ = _train_probe_head(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        embed_dim=embed_dim,
        device=device,
        epochs=epochs,
        probe_lr=probe_lr,
        weight_decay=weight_decay,
        num_classes=num_classes,
        cosine_schedule=False,
    )
    return head


def probe_model(
    model: IJEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    embed_dim: int,
    epochs: int = 20,
    probe_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_classes: int = 10,
) -> dict[str, float]:
    """Linear probe on a frozen encoder (fixed LR, features extracted once)."""
    was_training = model.training

    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)
    train_feats, val_feats, feat_std = _standardize_features(train_feats, val_feats)

    head, accuracy = _train_probe_head(
        train_feats,
        train_labels,
        val_feats,
        val_labels,
        embed_dim=embed_dim,
        device=device,
        epochs=epochs,
        probe_lr=probe_lr,
        weight_decay=weight_decay,
        num_classes=num_classes,
        cosine_schedule=False,
    )

    if was_training:
        model.train()

    head.eval()
    with torch.no_grad():
        preds = head(val_feats.to(device)).argmax(dim=1).cpu()
    correct = (preds == val_labels).sum().item()

    return {
        "top1_accuracy": accuracy,
        "correct": float(correct),
        "total": float(val_labels.numel()),
        "feat_std": feat_std,
    }


def probe_model_tuned(
    model: IJEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    embed_dim: int,
    epochs: int = 100,
    lr_grid: tuple[float, ...] = (3e-4, 1e-3, 3e-3),
    weight_decay: float = 1e-4,
    num_classes: int = 10,
    feat_std: float | None = None,
) -> dict[str, float]:
    """Linear probe with per-LR cosine schedule; reports the best val accuracy."""
    was_training = model.training

    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)
    train_feats, val_feats, computed_std = _standardize_features(train_feats, val_feats)
    if feat_std is None:
        feat_std = computed_std

    results_by_lr: dict[float, float] = {}
    for lr in lr_grid:
        _, acc = _train_probe_head(
            train_feats,
            train_labels,
            val_feats,
            val_labels,
            embed_dim=embed_dim,
            device=device,
            epochs=epochs,
            probe_lr=lr,
            weight_decay=weight_decay,
            num_classes=num_classes,
            cosine_schedule=True,
            seed=0,
        )
        results_by_lr[lr] = acc
        print(f"  [tuned-probe] lr={lr:.0e}  best_val_top1={acc:.2f}%")

    best_lr = max(results_by_lr, key=results_by_lr.get)
    best_acc = results_by_lr[best_lr]

    if was_training:
        model.train()

    return {
        "top1_accuracy": best_acc,
        "best_lr": best_lr,
        "results_by_lr": results_by_lr,
        "feat_std": feat_std,
    }


def load_encoder_from_checkpoint(
    cfg: dict[str, Any],
    checkpoint: str | Path,
    device: torch.device,
) -> IJEPA:
    """Load a checkpoint into a fresh model. Encoder weights are what matter for probing."""
    model = IJEPA.from_config(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        print(f"  [probe] ignoring unexpected keys: {unexpected}")
    if missing:
        print(f"  [probe] missing keys (left at init): {missing}")
    for p in model.encoder.parameters():
        p.requires_grad = False
    return model


def run_linear_probe(
    cfg: dict[str, Any],
    checkpoint: str | Path,
    device: torch.device | None = None,
    probe_epochs: int | None = None,
) -> dict[str, float]:
    device = device or get_device(cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 42))
    model = load_encoder_from_checkpoint(cfg, checkpoint, device)

    train_loader, val_loader = build_dataloaders(
        cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        train_augment=False,
    )

    return probe_model(
        model,
        train_loader,
        val_loader,
        device,
        embed_dim=cfg["encoder"]["embed_dim"],
        epochs=probe_epochs or cfg["eval"].get("probe_epochs", 20),
        probe_lr=cfg["eval"].get("probe_lr", 1e-3),
        weight_decay=cfg["eval"].get("probe_weight_decay", 1e-4),
    )


def run_linear_probe_tuned(
    cfg: dict[str, Any],
    checkpoint: str | Path,
    device: torch.device | None = None,
    epochs: int = 100,
    lr_grid: tuple[float, ...] = (3e-4, 1e-3, 3e-3),
) -> dict[str, Any]:
    device = device or get_device(cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 42))
    model = load_encoder_from_checkpoint(cfg, checkpoint, device)

    train_loader, val_loader = build_dataloaders(
        cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        train_augment=False,
    )

    return probe_model_tuned(
        model,
        train_loader,
        val_loader,
        device,
        embed_dim=cfg["encoder"]["embed_dim"],
        epochs=epochs,
        lr_grid=lr_grid,
        weight_decay=cfg["eval"].get("probe_weight_decay", 1e-4),
    )
