from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from jepa.data.cifar10 import build_dataloaders
from jepa.eval.linear_probe import probe_model, probe_model_tuned
from jepa.masking import IJEPAMaskCollator
from jepa.models.jepa import IJEPA
from jepa.models.looped_predictor import LoopedPredictor, expected_loops_from_exit_probs
from jepa.utils.logging import RunLogger
from jepa.utils.seed import set_seed


def ema_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    progress = min(1.0, step / max(1, total_steps))
    return start + progress * (end - start)


def ema_schedule_cosine(step: int, total_steps: int, start: float, end: float) -> float:
    """Cosine EMA schedule (DINO / MoCo-v3 style)."""
    progress = min(1.0, step / max(1, total_steps))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def ema_schedule_for(step: int, total_steps: int, start: float, end: float, kind: str) -> float:
    if kind == "cosine":
        return ema_schedule_cosine(step, total_steps, start, end)
    return ema_schedule(step, total_steps, start, end)


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def stack_indices(indices_list: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.stack(indices_list).to(device)


def _opt_range(value: Any) -> tuple[int, int] | None:
    """Coerce a config list/tuple into an (int, int) range, or ``None``."""
    if value is None:
        return None
    return (int(value[0]), int(value[1]))


@torch.no_grad()
def run_probe(
    model: IJEPA,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Fixed-LR linear probe for in-training monitoring."""
    eval_cfg = cfg.get("eval", {}) or {}
    probe_train, probe_val = build_dataloaders(
        cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        train_augment=False,
    )
    with torch.enable_grad():
        results = probe_model(
            model,
            probe_train,
            probe_val,
            device,
            embed_dim=cfg["encoder"]["embed_dim"],
            epochs=eval_cfg.get("probe_epochs", 20),
            probe_lr=eval_cfg.get("probe_lr", 1e-3),
            weight_decay=eval_cfg.get("probe_weight_decay", 1e-4),
        )
    return results


def run_probe_tuned(
    model: IJEPA,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Tuned linear probe (cosine LR + grid search) for final reporting."""
    eval_cfg = cfg.get("eval", {}) or {}
    probe_train, probe_val = build_dataloaders(
        cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        train_augment=False,
    )
    lr_grid = tuple(eval_cfg.get("tuned_lr_grid", (3e-4, 1e-3, 3e-3)))
    with torch.enable_grad():
        results = probe_model_tuned(
            model,
            probe_train,
            probe_val,
            device,
            embed_dim=cfg["encoder"]["embed_dim"],
            epochs=eval_cfg.get("tuned_probe_epochs", 100),
            lr_grid=lr_grid,
            weight_decay=eval_cfg.get("probe_weight_decay", 1e-4),
        )
    return results


def train_epoch(
    model: IJEPA,
    loader: DataLoader,
    mask_collator: IJEPAMaskCollator,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    global_step: int,
    total_steps: int,
    cfg: dict[str, Any],
    logger: RunLogger | None = None,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    train_cfg = cfg["train"]
    beta = train_cfg.get("exit_entropy_beta", 0.01)
    reg_cfg = cfg.get("regularizer", {}) or {}
    reg_warmup = reg_cfg.get("warmup_steps", 0)
    two_view = bool(cfg.get("two_view", False))
    ema_kind = train_cfg.get("ema_schedule", "linear")
    ema_start = train_cfg.get("ema_momentum_start", 0.996)
    ema_end = train_cfg.get("ema_momentum_end", 1.0)

    for batch in tqdm(loader, desc="train", leave=False):
        if two_view:
            (strong, weak), _ = batch
            strong = strong.to(device)
            weak = weak.to(device)
            images = strong
            teacher_images = weak
        else:
            images, _ = batch
            images = images.to(device)
            teacher_images = None

        masks = mask_collator(images.shape[0])
        context_indices = stack_indices(masks.context_indices, device)
        target_indices = stack_indices(masks.target_indices, device)

        if getattr(model, "reg_enabled", False) and reg_warmup > 0:
            model.reg_scale = min(1.0, global_step / reg_warmup)

        out = model(images, context_indices, target_indices, teacher_images=teacher_images)
        loss = out["loss"]

        if isinstance(model.predictor, LoopedPredictor) and "exit_probs" in out:
            loss = model.predictor.compute_total_loss(loss, out["exit_probs"], beta=beta)

        optimizer.zero_grad()
        loss.backward()
        if train_cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        scheduler.step()

        momentum = ema_schedule_for(global_step, total_steps, ema_start, ema_end, ema_kind)
        model.update_target_encoder(momentum)

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        if logger and global_step % train_cfg.get("log_every", 50) == 0:
            metrics = {"train/loss": loss.item(), "train/ema_momentum": momentum}
            if "pred_loss" in out:
                metrics["train/pred_loss"] = out["pred_loss"].item()
            if "exit_probs" in out:
                exit_probs = out["exit_probs"]
                metrics["train/exit_prob_mean"] = float(exit_probs.mean().item())
                for loop_i in range(exit_probs.shape[1]):
                    metrics[f"train/exit_prob_loop_{loop_i + 1}"] = float(
                        exit_probs[:, loop_i].mean().item()
                    )
                metrics["train/expected_loops"] = float(
                    expected_loops_from_exit_probs(exit_probs).mean().item()
                )
            if "reg_var" in out:
                metrics["train/reg_var"] = out["reg_var"].item()
                metrics["train/reg_cov"] = out["reg_cov"].item()
            logger.log(metrics, global_step)

    return total_loss / max(1, num_batches), global_step


def save_checkpoint(
    path: Path,
    model: IJEPA,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    step: int,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: IJEPA,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
) -> dict[str, Any]:
    """Restore model, optimizer, and scheduler from a training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    elif "step" in ckpt:
        # Older checkpoints saved before scheduler state was included.
        for _ in range(int(ckpt["step"])):
            scheduler.step()
    return ckpt


def train(
    cfg: dict[str, Any],
    device: torch.device,
    resume_from: str | Path | None = None,
) -> Path:
    if resume_from is None:
        set_seed(cfg.get("seed", 42))

    two_view = bool(cfg.get("two_view", False))
    train_loader, _ = build_dataloaders(
        cfg["data"]["data_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        train_augment=True,
        augmentation=cfg.get("augmentation"),
        two_view=two_view,
    )

    model = IJEPA.from_config(cfg).to(device)
    train_cfg = cfg["train"]
    print(f"Trainable parameters: {model.num_trainable_params():,}")
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.05),
    )

    epochs = train_cfg["epochs"]
    total_steps = epochs * len(train_loader)
    # Prefer warmup_epochs; fall back to a fixed step count.
    if "warmup_epochs" in train_cfg:
        warmup_steps = int(train_cfg["warmup_epochs"] * len(train_loader))
    else:
        warmup_steps = train_cfg.get("warmup_steps", 500)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    grid_size = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    mask_cfg = cfg["masking"]
    mask_collator = IJEPAMaskCollator(
        grid_size=grid_size,
        num_target_blocks=mask_cfg.get("num_target_blocks", 4),
        target_scale=tuple(mask_cfg.get("target_scale", [0.15, 0.2])),
        context_scale=tuple(mask_cfg.get("context_scale", [0.85, 1.0])),
        fixed_context_patches=mask_cfg.get("fixed_context_patches", 32),
        fixed_target_patches=mask_cfg.get("fixed_target_patches", 16),
        context_patches_range=_opt_range(mask_cfg.get("context_patches_range")),
        target_patches_range=_opt_range(mask_cfg.get("target_patches_range")),
    )

    run_dir = Path(train_cfg.get("run_dir", "runs/default"))
    logger = RunLogger(run_dir, use_wandb=train_cfg.get("use_wandb", False))
    logger.init(cfg)

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    global_step = 0
    start_epoch = 0
    probe_every = (cfg.get("eval", {}) or {}).get("probe_every_epochs", 0)

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        print(f"Resumed from {resume_path} (epoch {start_epoch}/{epochs}, step {global_step})")

    for epoch in range(start_epoch, epochs):
        avg_loss, global_step = train_epoch(
            model,
            train_loader,
            mask_collator,
            optimizer,
            scheduler,
            device,
            global_step,
            total_steps,
            cfg,
            logger,
        )
        print(f"epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
        logger.log({"epoch": epoch + 1, "train/epoch_loss": avg_loss}, global_step)
        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, epoch + 1, global_step, cfg)

        is_final = epoch + 1 == epochs
        if probe_every and ((epoch + 1) % probe_every == 0 or is_final):
            res = run_probe(model, cfg, device)
            acc, feat_std = res["top1_accuracy"], res["feat_std"]
            print(f"  [probe] epoch {epoch + 1}  top1={acc:.2f}%  feat_std={feat_std:.4f}")
            logger.log({"eval/probe_top1": acc, "eval/feat_std": feat_std}, global_step)

        if is_final:
            print("  [tuned-probe] running final tuned linear probe (LR sweep)...")
            tuned = run_probe_tuned(model, cfg, device)
            final_acc = tuned["top1_accuracy"]
            print(
                f"  [tuned-probe] FINAL top1={final_acc:.2f}%  "
                f"best_lr={tuned['best_lr']:.0e}  feat_std={tuned['feat_std']:.4f}"
            )
            logger.log(
                {
                    "eval/final_top1": final_acc,
                    "eval/final_best_lr": tuned["best_lr"],
                    "eval/final_feat_std": tuned["feat_std"],
                },
                global_step,
            )

    return ckpt_dir / "latest.pt"
