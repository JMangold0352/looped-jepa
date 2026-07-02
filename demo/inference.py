"""Model loading + demo inference for the baseline vs looped comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from jepa.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD
from jepa.masking import IJEPAMaskCollator
from jepa.models.jepa import IJEPA
from jepa.models.looped_predictor import LoopedPredictor, expected_loops_from_exit_probs
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.seed import set_seed
from visualizations.inference import capture_block_attention

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

BASELINE = {
    "label": "v3 baseline",
    "config": "configs/image_jepa_cifar10_v3.yaml",
    "checkpoint": "checkpoints/baseline_v3/latest.pt",
}
LOOPED = {
    "label": "v3 looped",
    "config": "configs/image_jepa_cifar10_v3_looped.yaml",
    "checkpoint": "checkpoints/baseline_v3_looped/latest.pt",
}


@dataclass
class DemoContext:
    device: torch.device
    baseline: IJEPA
    looped: IJEPA
    collator: IJEPAMaskCollator
    transform: T.Compose
    grid_size: int
    probe_heads: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    message: str = ""


def _load_model(spec: dict, device: torch.device) -> IJEPA:
    cfg = load_config(str(PROJECT_ROOT / spec["config"]))
    model = IJEPA.from_config(cfg).to(device)
    ckpt = torch.load(PROJECT_ROOT / spec["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def load_context() -> DemoContext:
    """Load both models once. Degrades gracefully if checkpoints are missing."""
    device = get_device("auto")
    cfg = load_config(str(PROJECT_ROOT / BASELINE["config"]))
    set_seed(cfg.get("seed", 42))
    img_size = cfg["data"]["img_size"]
    grid_size = img_size // cfg["data"]["patch_size"]
    collator = IJEPAMaskCollator(
        grid_size=grid_size,
        fixed_context_patches=cfg["masking"].get("fixed_context_patches", 32),
        fixed_target_patches=cfg["masking"].get("fixed_target_patches", 16),
    )
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    missing = [s["checkpoint"] for s in (BASELINE, LOOPED) if not (PROJECT_ROOT / s["checkpoint"]).exists()]
    if missing:
        return DemoContext(
            device=device, baseline=None, looped=None, collator=collator,
            transform=transform, grid_size=grid_size, available=False,
            message="Checkpoints not found: " + ", ".join(missing),
        )

    return DemoContext(
        device=device,
        baseline=_load_model(BASELINE, device),
        looped=_load_model(LOOPED, device),
        collator=collator,
        transform=transform,
        grid_size=grid_size,
    )


def _sample_mask(ctx: DemoContext, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    masks = ctx.collator(1)
    c = masks.context_indices[0].unsqueeze(0).to(ctx.device)
    t = masks.target_indices[0].unsqueeze(0).to(ctx.device)
    return c, t


@torch.no_grad()
def _run_stack_capture(base, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the shared block stack once, capturing the last block's attention."""
    blocks = base.block_stack.blocks
    attn_last = None
    for i, block in enumerate(blocks):
        if i == len(blocks) - 1:
            attn_last = capture_block_attention(block, x)
        x = block(x)
    x = base.norm(x)
    return x, attn_last


def _attn_grid(attn: torch.Tensor, ctx_idx: torch.Tensor, n_ctx: int, grid_size: int) -> np.ndarray:
    """Mean attention that context patches receive from target tokens, on the image grid."""
    tgt_to_ctx = attn[:, :, n_ctx:, :n_ctx].mean(dim=1).mean(dim=1)[0].cpu().numpy()  # (n_ctx,)
    out = np.zeros(grid_size * grid_size, dtype=np.float32)
    for pos, idx in enumerate(ctx_idx[0].cpu().tolist()):
        out[int(idx)] = tgt_to_ctx[pos]
    return out.reshape(grid_size, grid_size)


@torch.no_grad()
def _teacher_targets(model: IJEPA, images: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    teacher = model.target_encoder.forward_all_patches(images)
    idx = tgt.unsqueeze(-1).expand(-1, -1, teacher.size(-1))
    return torch.gather(teacher, 1, idx)


@torch.no_grad()
def run_model(
    model: IJEPA,
    images: torch.Tensor,
    ctx: torch.Tensor,
    tgt: torch.Tensor,
    grid_size: int,
    loops: int,
) -> dict[str, Any]:
    """Per-loop cosine, attention grids, exit stats, and final per-patch cosine."""
    predictor = model.predictor
    is_looped = isinstance(predictor, LoopedPredictor)
    base = predictor.base_predictor if is_looped else predictor
    n_loops = loops if is_looped else 1

    context_repr = model.encoder(images, ctx)
    target_repr = _teacher_targets(model, images, tgt)
    x = base.build_sequence(context_repr, ctx, tgt)
    n_ctx = ctx.shape[1]

    per_loop_mean: list[float] = []
    attn_grids: list[np.ndarray] = []
    exit_logits: list[torch.Tensor] = []
    cos_per_patch = None

    for _ in range(n_loops):
        x, attn = _run_stack_capture(base, x)
        pred = base.output_proj(x[:, n_ctx:])
        cos = F.cosine_similarity(pred, target_repr, dim=-1)[0]  # (n_tgt,)
        per_loop_mean.append(float(cos.mean().item()))
        cos_per_patch = cos.cpu().numpy()
        attn_grids.append(_attn_grid(attn, ctx, n_ctx, grid_size))
        if is_looped and predictor.use_exit_gate and predictor.exit_gate is not None:
            exit_logits.append(torch.sigmoid(predictor.exit_gate(x.mean(dim=1))).squeeze(-1))

    result: dict[str, Any] = {
        "looped": is_looped,
        "per_loop_cosine": per_loop_mean,
        "final_cosine": per_loop_mean[-1],
        "cos_per_patch": cos_per_patch,
        "attn_grids": attn_grids,
        "exit_probs": None,
        "expected_loops": None,
    }
    if exit_logits:
        probs = torch.stack(exit_logits, dim=1)  # (1, n_loops)
        result["exit_probs"] = probs[0].cpu().tolist()
        result["expected_loops"] = float(expected_loops_from_exit_probs(probs)[0].item())
    return result


def get_probe_head(ctx: DemoContext, key: str) -> Any:
    """Lazily fit + cache a linear probe head for CIFAR-10 class predictions."""
    if key in ctx.probe_heads:
        return ctx.probe_heads[key]
    from jepa.data.cifar10 import build_dataloaders
    from jepa.eval.linear_probe import train_probe_head

    model = ctx.baseline if key == "baseline" else ctx.looped
    cfg = load_config(str(PROJECT_ROOT / (BASELINE if key == "baseline" else LOOPED)["config"]))
    train_loader, val_loader = build_dataloaders(
        str(PROJECT_ROOT / cfg["data"]["data_dir"]),
        batch_size=cfg["data"]["batch_size"],
        num_workers=0,
        train_augment=False,
    )
    head = train_probe_head(
        model, train_loader, val_loader, ctx.device,
        embed_dim=cfg["encoder"]["embed_dim"],
        epochs=10,
        probe_lr=cfg["eval"].get("probe_lr", 1e-3),
        weight_decay=cfg["eval"].get("probe_weight_decay", 1e-4),
    )
    ctx.probe_heads[key] = head
    return head


@torch.no_grad()
def probe_predict(ctx: DemoContext, key: str, images: torch.Tensor, topk: int = 3) -> list[tuple[str, float]]:
    model = ctx.baseline if key == "baseline" else ctx.looped
    head = get_probe_head(ctx, key)
    tokens = model.encoder.forward_all_patches(images)
    logits = head(tokens.mean(dim=1))
    probs = torch.softmax(logits, dim=-1)[0].cpu()
    top = probs.topk(topk)
    return [(CIFAR10_CLASSES[i], float(p)) for p, i in zip(top.values.tolist(), top.indices.tolist())]
