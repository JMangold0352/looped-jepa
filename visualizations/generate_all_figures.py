#!/usr/bin/env python3
"""Generate publication-ready figures for v3 baseline vs looped predictor.

Produces PNG + PDF at 300 DPI under ``visualizations/figures/`` and per-loop
diagnostics under ``visualizations/loop_analysis/``.

Usage::

    python visualizations/generate_all_figures.py           # full suite
    python visualizations/generate_all_figures.py --fast    # smoke test
    python visualizations/generate_all_figures.py --loop-analysis-only

Requires checkpoints at default paths (see ``--baseline-checkpoint`` / ``--looped-checkpoint``).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from jepa.data.cifar10 import build_dataloaders
from jepa.eval.linear_probe import extract_features, load_encoder_from_checkpoint
from jepa.masking import IJEPAMaskCollator, MaskBatch
from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.seed import set_seed
from visualizations.figures.ablation_summary import plot_ablation_summary
from visualizations.figures.attention_maps import plot_attention_comparison
from visualizations.figures.embeddings import plot_embedding_comparison, plot_per_loop_cosine
from visualizations.figures.loop_analysis import generate_all_loop_analysis
from visualizations.figures.mask_reconstruction import plot_mask_reconstruction_comparison
from visualizations.figures.training_curves import (
    plot_exit_loop_distribution,
    plot_expected_loops_training,
    plot_training_evaluation_curves,
)
from visualizations.inference import collect_loop_usage
from visualizations.loop_collect import collect_loop_sample_records, summarize_loop_records
from visualizations.style import apply_style

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _load_full_model(cfg_path: str, checkpoint: str, device: torch.device) -> IJEPA:
    cfg = load_config(cfg_path)
    model = IJEPA.from_config(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def _slice_masks(masks: MaskBatch, n: int) -> MaskBatch:
    return MaskBatch(
        context_indices=masks.context_indices[:n],
        target_indices=masks.target_indices[:n],
    )


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all publication figures")
    parser.add_argument("--baseline-config", default="configs/image_jepa_cifar10_v3.yaml")
    parser.add_argument("--looped-config", default="configs/image_jepa_cifar10_v3_looped.yaml")
    parser.add_argument("--baseline-checkpoint", default="checkpoints/baseline_v3/latest.pt")
    parser.add_argument("--looped-checkpoint", default="checkpoints/baseline_v3_looped/latest.pt")
    parser.add_argument("--baseline-metrics", default="runs/cifar10_baseline_v3/metrics.jsonl")
    parser.add_argument("--looped-metrics", default="runs/cifar10_v3_looped/metrics.jsonl")
    parser.add_argument("--ablation-summary", default="results/ablations/summary.json")
    parser.add_argument("--out-dir", default="visualizations/figures")
    parser.add_argument("--fast", action="store_true", help="Small sample counts for smoke tests")
    parser.add_argument("--embed-method", choices=("tsne", "pca"), default="tsne")
    parser.add_argument("--loop-analysis-dir", default="visualizations/loop_analysis")
    parser.add_argument(
        "--loop-analysis-only",
        action="store_true",
        help="Only run per-loop deep-dive figures (Prompt 3)",
    )
    parser.add_argument("--skip-loop-analysis", action="store_true")
    args = parser.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fast:
        n_recon = 2
        n_attn = 2
        max_embed = 400
        cosine_batches = 4
        loop_usage_batches = 8
        loop_analysis_batches = 16
        batch_size = 32
    else:
        n_recon = 6
        n_attn = 3
        max_embed = 2000
        cosine_batches = 32
        loop_usage_batches = None
        loop_analysis_batches = None
        batch_size = 64

    device = get_device("auto")
    set_seed(42)
    t0 = time.time()
    _log(f"Device: {device}")

    baseline_cfg = load_config(args.baseline_config)
    grid_size = baseline_cfg["data"]["img_size"] // baseline_cfg["data"]["patch_size"]

    _, val_loader = build_dataloaders(
        baseline_cfg["data"]["data_dir"],
        batch_size=batch_size,
        num_workers=0,
        train_augment=False,
    )
    collator = IJEPAMaskCollator(
        grid_size=grid_size,
        fixed_context_patches=baseline_cfg["masking"].get("fixed_context_patches", 32),
        fixed_target_patches=baseline_cfg["masking"].get("fixed_target_patches", 16),
    )

    images_list, labels_list = [], []
    seen_classes: set[int] = set()
    for images, labels in val_loader:
        for i in range(images.shape[0]):
            cls = int(labels[i].item())
            if cls in seen_classes:
                continue
            seen_classes.add(cls)
            images_list.append(images[i])
            labels_list.append(labels[i])
            if len(images_list) >= max(n_recon, n_attn, 10):
                break
        if len(images_list) >= max(n_recon, n_attn, 10):
            break

    example_images = torch.stack(images_list)
    example_labels = torch.stack(labels_list)
    example_masks = collator(example_images.shape[0])

    _log("Loading models...")
    baseline_cfg_dict = load_config(args.baseline_config)
    looped_cfg_dict = load_config(args.looped_config)
    baseline_encoder = load_encoder_from_checkpoint(baseline_cfg_dict, args.baseline_checkpoint, device)
    looped_encoder = load_encoder_from_checkpoint(looped_cfg_dict, args.looped_checkpoint, device)
    baseline_model = _load_full_model(args.baseline_config, args.baseline_checkpoint, device)
    looped_model = _load_full_model(args.looped_config, args.looped_checkpoint, device)

    loop_analysis_dir = Path(args.loop_analysis_dir)

    if args.loop_analysis_only:
        _log("[loop] Per-loop deep dive (Prompt 3)...")
        records, record_images = collect_loop_sample_records(
            looped_model, val_loader, collator, device, max_batches=loop_analysis_batches
        )
        summary = summarize_loop_records(records)
        loop_analysis_dir.mkdir(parents=True, exist_ok=True)
        (loop_analysis_dir / "summary.json").write_text(
            __import__("json").dumps(summary, indent=2)
        )
        generate_all_loop_analysis(records, record_images, list(CIFAR10_CLASSES), loop_analysis_dir)
        _log(f"Loop analysis saved to {loop_analysis_dir.resolve()}")
        return

    _log("[1/6] Mask reconstruction panels...")
    plot_mask_reconstruction_comparison(
        baseline_model,
        looped_model,
        example_images[:n_recon],
        example_labels[:n_recon],
        _slice_masks(example_masks, n_recon),
        grid_size,
        list(CIFAR10_CLASSES),
        out_dir / "01_mask_reconstruction",
        device=device,
    )

    _log("[2/6] Predictor attention maps...")
    plot_attention_comparison(
        baseline_model,
        looped_model,
        example_images[:n_attn],
        example_labels[:n_attn],
        _slice_masks(example_masks, n_attn),
        grid_size,
        list(CIFAR10_CLASSES),
        out_dir / "02_attention_maps",
        n_examples=n_attn,
        device=device,
    )

    _log("[3/6] Embedding visualizations...")
    base_feats, labels = extract_features(baseline_encoder, val_loader, device)
    loop_feats, _ = extract_features(looped_encoder, val_loader, device)
    plot_embedding_comparison(
        base_feats,
        loop_feats,
        labels,
        out_dir / "03_embeddings",
        max_points=max_embed,
        method=args.embed_method,
    )
    plot_per_loop_cosine(
        looped_model,
        val_loader,
        collator,
        device,
        out_dir / "03_per_loop_cosine",
        max_batches=cosine_batches,
    )

    _log("[4/6] Training & evaluation curves...")
    if Path(args.baseline_metrics).exists() and Path(args.looped_metrics).exists():
        plot_training_evaluation_curves(
            Path(args.baseline_metrics),
            Path(args.looped_metrics),
            out_dir / "04_training_curves",
        )
        plot_expected_loops_training(Path(args.looped_metrics), out_dir / "04_expected_loops_training")
    loop_usage = collect_loop_usage(looped_model, val_loader, collator, device, max_batches=loop_usage_batches)
    plot_exit_loop_distribution(loop_usage, out_dir / "04_exit_loop_distribution")

    _log("[5/6] Ablation summary...")
    if Path(args.ablation_summary).exists():
        plot_ablation_summary(Path(args.ablation_summary), out_dir / "05_ablation_summary")

    if not args.skip_loop_analysis:
        _log("[6/6] Per-loop deep dive...")
        records, record_images = collect_loop_sample_records(
            looped_model, val_loader, collator, device, max_batches=loop_analysis_batches
        )
        summary = summarize_loop_records(records)
        loop_analysis_dir.mkdir(parents=True, exist_ok=True)
        (loop_analysis_dir / "summary.json").write_text(
            __import__("json").dumps(summary, indent=2)
        )
        generate_all_loop_analysis(records, record_images, list(CIFAR10_CLASSES), loop_analysis_dir)
        _log(f"Loop analysis saved to {loop_analysis_dir.resolve()}")

    elapsed = time.time() - t0
    _log(f"Done in {elapsed / 60:.1f} min. Figures saved to {out_dir.resolve()}")
    _log("Each figure written as .png and .pdf at 300 DPI.")


if __name__ == "__main__":
    main()
