from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from jepa.masking import IJEPAMaskCollator
from jepa.models.jepa import IJEPA
from jepa.models.looped_predictor import LoopedPredictor, expected_loops_from_exit_probs
from jepa.train import stack_indices


@torch.no_grad()
def measure_loop_usage(
    model: IJEPA,
    val_loader: DataLoader,
    mask_collator: IJEPAMaskCollator,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Summarize exit-gate behaviour and effective loop depth on a val loader."""
    model.eval()
    predictor = model.predictor
    if not isinstance(predictor, LoopedPredictor):
        return {
            "max_loops": 1,
            "use_exit_gate": False,
            "mean_loops_used": 1.0,
            "exit_prob_per_loop": [],
            "loops_used_histogram": {1: int(len(val_loader.dataset))},
        }

    max_loops = predictor.max_loops
    all_exit_probs: list[torch.Tensor] = []
    all_expected: list[torch.Tensor] = []

    for batch_idx, (images, _) in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        masks = mask_collator(images.shape[0])
        context_indices = stack_indices(masks.context_indices, device)
        target_indices = stack_indices(masks.target_indices, device)

        context_repr = model.encoder(images, context_indices)
        pred_out = predictor(context_repr, context_indices, target_indices)

        if isinstance(pred_out, tuple):
            _, exit_probs = pred_out
            all_exit_probs.append(exit_probs.detach().cpu())
            all_expected.append(expected_loops_from_exit_probs(exit_probs).cpu())
        else:
            batch = images.shape[0]
            all_expected.append(torch.full((batch,), float(max_loops)))

    if all_expected:
        expected = torch.cat(all_expected)
        mean_loops = float(expected.mean().item())
        hist: dict[int, int] = {}
        for loops in expected.round().to(torch.int64).tolist():
            hist[int(loops)] = hist.get(int(loops), 0) + 1
    else:
        mean_loops = float(max_loops)
        hist = {max_loops: 0}

    exit_prob_per_loop: list[float] = []
    if all_exit_probs:
        stacked = torch.cat(all_exit_probs, dim=0)
        exit_prob_per_loop = stacked.mean(dim=0).tolist()

    return {
        "max_loops": max_loops,
        "use_exit_gate": bool(predictor.use_exit_gate),
        "mean_loops_used": mean_loops,
        "std_loops_used": float(expected.std().item()) if all_expected else 0.0,
        "exit_prob_per_loop": exit_prob_per_loop,
        "loops_used_histogram": hist,
        "num_samples": int(expected.numel()) if all_expected else 0,
    }


def training_stability_from_metrics(metrics_path: str) -> dict[str, float]:
    """Derive simple stability stats from a ``metrics.jsonl`` training log."""
    import json
    from pathlib import Path

    path = Path(metrics_path)
    if not path.exists():
        return {}

    epoch_losses: list[float] = []
    feat_stds: list[float] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "train/epoch_loss" in record:
            epoch_losses.append(float(record["train/epoch_loss"]))
        if "eval/feat_std" in record:
            feat_stds.append(float(record["eval/feat_std"]))

    if not epoch_losses:
        return {}

    tail = epoch_losses[-25:] if len(epoch_losses) >= 25 else epoch_losses
    return {
        "final_epoch_loss": epoch_losses[-1],
        "tail_loss_mean": sum(tail) / len(tail),
        "tail_loss_std": float(torch.tensor(tail).std().item()) if len(tail) > 1 else 0.0,
        "min_feat_std": min(feat_stds) if feat_stds else 0.0,
        "final_feat_std_probe": feat_stds[-1] if feat_stds else 0.0,
    }
