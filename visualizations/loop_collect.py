"""Per-sample loop metrics for deep-dive visualizations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from jepa.models.jepa import IJEPA
from jepa.models.looped_predictor import LoopedPredictor, expected_loops_from_exit_probs


@dataclass
class LoopSampleRecord:
    """Per-image loop behaviour on one mask draw."""

    label: int
    expected_loops: float
    exit_probs: list[float] = field(default_factory=list)
    cosine_by_loop: list[float] = field(default_factory=list)
    l1_by_loop: list[float] = field(default_factory=list)

    @property
    def loops_rounded(self) -> int:
        return int(round(self.expected_loops))

    @property
    def cosine_gain(self) -> float:
        if len(self.cosine_by_loop) < 2:
            return 0.0
        return self.cosine_by_loop[-1] - self.cosine_by_loop[0]


@torch.no_grad()
def per_sample_loop_metrics_batch(
    model: IJEPA,
    images: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> list[LoopSampleRecord]:
    """Compute per-image exit depth and per-loop prediction quality."""
    model.eval()
    predictor = model.predictor
    b = images.shape[0]
    context_repr = model.encoder(images, context_indices)
    teacher_tokens = model.target_encoder.forward_all_patches(images)
    idx = target_indices.unsqueeze(-1).expand(-1, -1, teacher_tokens.size(-1))
    target_repr = torch.gather(teacher_tokens, 1, idx)

    records: list[LoopSampleRecord] = []

    if not isinstance(predictor, LoopedPredictor):
        pred = predictor(context_repr, context_indices, target_indices)
        cos = F.cosine_similarity(pred, target_repr, dim=-1).mean(dim=-1)
        l1 = F.smooth_l1_loss(pred, target_repr, reduction="none").mean(dim=(1, 2))
        for i in range(b):
            records.append(
                LoopSampleRecord(
                    label=-1,
                    expected_loops=1.0,
                    cosine_by_loop=[float(cos[i])],
                    l1_by_loop=[float(l1[i])],
                )
            )
        return records

    base = predictor.base_predictor
    x = base.build_sequence(context_repr, context_indices, target_indices)
    n_ctx = context_indices.shape[1]
    max_loops = predictor.max_loops

    cosine_steps: list[torch.Tensor] = []
    l1_steps: list[torch.Tensor] = []
    exit_probs_tensor: torch.Tensor | None = None

    exit_list: list[torch.Tensor] = []
    for loop_i in range(max_loops):
        x = base.forward_stack(x)
        pred = base.output_proj(x[:, n_ctx:])
        cos = F.cosine_similarity(pred, target_repr, dim=-1).mean(dim=-1)
        l1 = F.smooth_l1_loss(pred, target_repr, reduction="none").mean(dim=(1, 2))
        cosine_steps.append(cos)
        l1_steps.append(l1)
        if predictor.use_exit_gate and predictor.exit_gate is not None:
            gate_in = x.mean(dim=1)
            exit_list.append(torch.sigmoid(predictor.exit_gate(gate_in)).squeeze(-1))

    if exit_list:
        exit_probs_tensor = torch.stack(exit_list, dim=1)
        expected = expected_loops_from_exit_probs(exit_probs_tensor)
    else:
        expected = torch.full((b,), float(max_loops), device=images.device)

    for i in range(b):
        ep = exit_probs_tensor[i].tolist() if exit_probs_tensor is not None else []
        records.append(
            LoopSampleRecord(
                label=-1,
                expected_loops=float(expected[i].item()),
                exit_probs=[float(v) for v in ep],
                cosine_by_loop=[float(c[i]) for c in cosine_steps],
                l1_by_loop=[float(v[i]) for v in l1_steps],
            )
        )
    return records


@torch.no_grad()
def collect_loop_sample_records(
    model: IJEPA,
    val_loader,
    mask_collator,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[list[LoopSampleRecord], list[torch.Tensor]]:
    """Walk the val loader and collect per-sample metrics plus images for qual panels."""
    records: list[LoopSampleRecord] = []
    images_store: list[torch.Tensor] = []

    for batch_idx, (images, labels) in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        masks = mask_collator(images.shape[0])
        ctx = torch.stack(masks.context_indices).to(device)
        tgt = torch.stack(masks.target_indices).to(device)
        batch_records = per_sample_loop_metrics_batch(model, images, ctx, tgt)
        for i, rec in enumerate(batch_records):
            rec.label = int(labels[i].item())
            records.append(rec)
            images_store.append(images[i].detach().cpu())
    return records, images_store


def summarize_loop_records(records: list[LoopSampleRecord]) -> dict[str, Any]:
    """Aggregate per-sample records for JSON sidecar export."""
    if not records:
        return {}
    expected = torch.tensor([r.expected_loops for r in records])
    hist: dict[int, int] = {}
    for v in expected.round().to(torch.int64).tolist():
        hist[int(v)] = hist.get(int(v), 0) + 1
    exit_prob_mean: list[float] = []
    if records[0].exit_probs:
        n_loops = len(records[0].exit_probs)
        for j in range(n_loops):
            exit_prob_mean.append(float(torch.tensor([r.exit_probs[j] for r in records]).mean()))
    return {
        "num_samples": len(records),
        "mean_loops_used": float(expected.mean()),
        "std_loops_used": float(expected.std()),
        "loops_used_histogram": hist,
        "exit_prob_per_loop": exit_prob_mean,
        "mean_cosine_gain": float(torch.tensor([r.cosine_gain for r in records]).mean()),
    }
