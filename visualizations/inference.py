"""Model inference helpers for visualization (attention, per-loop predictions)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from jepa.models.jepa import IJEPA
from jepa.models.looped_predictor import LoopedPredictor
from jepa.models.predictor import VisionTransformerPredictor
from jepa.models.vit import Block
from jepa.train import stack_indices


@torch.no_grad()
def capture_block_attention(block: Block, x: torch.Tensor) -> torch.Tensor:
    """Return attention weights (B, H, N, N) for one transformer block."""
    norm_x = block.norm1(x)
    attn = block.attn
    b, n, c = norm_x.shape
    qkv = attn.qkv(norm_x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, _v = qkv[0], qkv[1], qkv[2]
    if attn.rope is not None:
        q, k = attn.rope.apply(q, k)
    weights = (q @ k.transpose(-2, -1)) * attn.scale
    return weights.softmax(dim=-1)


@torch.no_grad()
def predictor_attention_maps(
    model: IJEPA,
    images: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    layer: str = "last",
) -> dict[str, Any]:
    """Attention from target tokens to all sequence positions (per loop if looped)."""
    model.eval()
    context_repr = model.encoder(images, context_indices)
    predictor = model.predictor

    def _maps_from_stack(
        base: VisionTransformerPredictor,
        x: torch.Tensor,
        n_ctx: int,
        loops: int,
    ) -> list[torch.Tensor]:
        maps: list[torch.Tensor] = []
        blocks = base.block_stack.blocks
        block_idx = -1 if layer == "last" else 0
        for _ in range(loops):
            for i, block in enumerate(blocks):
                if i == (len(blocks) - 1 if layer == "last" else 0):
                    attn = capture_block_attention(block, x)
                    # Per target token: mean attention mass on context patches (heads averaged).
                    tgt_attn = attn[:, :, n_ctx:, :n_ctx].mean(dim=1)  # (B, n_tgt, n_ctx)
                    maps.append(tgt_attn.cpu())
                x = block(x)
            x = base.norm(x)
        return maps

    if isinstance(predictor, LoopedPredictor):
        base = predictor.base_predictor
        x = base.build_sequence(context_repr, context_indices, target_indices)
        n_ctx = context_indices.shape[1]
        loop_maps = _maps_from_stack(base, x, n_ctx, predictor.max_loops)
        return {"maps": loop_maps, "max_loops": predictor.max_loops, "looped": True}

    base = predictor
    x = base.build_sequence(context_repr, context_indices, target_indices)
    n_ctx = context_indices.shape[1]
    loop_maps = _maps_from_stack(base, x, n_ctx, 1)
    return {"maps": loop_maps, "max_loops": 1, "looped": False}


@torch.no_grad()
def forward_jepa_batch(
    model: IJEPA,
    images: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Single forward pass returning predicted and teacher target representations."""
    if device is None:
        device = images.device
    images = images.to(device)
    context_indices = context_indices.to(device)
    target_indices = target_indices.to(device)
    out = model(images, context_indices, target_indices)
    return {
        "pred_repr": out["pred_repr"],
        "target_repr": out["target_repr"],
        "exit_probs": out.get("exit_probs"),
    }


@torch.no_grad()
def predict_per_loop_cosine(
    model: IJEPA,
    images: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> list[float]:
    """Batch mean cosine similarity (pred vs teacher target) after each loop."""
    model.eval()
    predictor = model.predictor
    context_repr = model.encoder(images, context_indices)
    teacher_tokens = model.target_encoder.forward_all_patches(images)
    idx = target_indices.unsqueeze(-1).expand(-1, -1, teacher_tokens.size(-1))
    target_repr = torch.gather(teacher_tokens, 1, idx)

    if not isinstance(predictor, LoopedPredictor):
        pred = predictor(context_repr, context_indices, target_indices)
        sim = F.cosine_similarity(pred, target_repr, dim=-1).mean().item()
        return [sim]

    base = predictor.base_predictor
    x = base.build_sequence(context_repr, context_indices, target_indices)
    n_ctx = context_indices.shape[1]
    sims: list[float] = []
    for _ in range(predictor.max_loops):
        x = base.forward_stack(x)
        pred = base.output_proj(x[:, n_ctx:])
        sim = F.cosine_similarity(pred, target_repr, dim=-1).mean().item()
        sims.append(sim)
    return sims


@torch.no_grad()
def collect_loop_usage(
    model: IJEPA,
    val_loader,
    mask_collator,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Per-sample expected loop counts for histogram figures."""
    from jepa.eval.loop_metrics import measure_loop_usage

    return measure_loop_usage(model, val_loader, mask_collator, device, max_batches=max_batches)


def stack_batch_masks(masks, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack collator mask indices for a batch."""
    return stack_indices(masks.context_indices, device), stack_indices(masks.target_indices, device)
