#!/usr/bin/env python3
"""Transfer learning: frozen JEPA encoders vs scratch ResNet18 on aerial imagery.

Compares v3 baseline and looped checkpoints with a tuned linear probe on a transfer
dataset. Default proxy: EuroSAT (no API key). Roboflow maritime supported with
``--download``.

Usage::

    # EuroSAT proxy (recommended for quick repro)
    python scripts/transfer_roboflow.py --source eurosat

    # Roboflow Aerial Maritime Drone
    export ROBOFLOW_API_KEY="..."
    python scripts/transfer_roboflow.py --download \\
        --workspace demm --project aerial-maritime-drone-dataset --version 1 \\
        --data-dir data/transfer/aerial_maritime

Outputs: ``results/transfer/transfer_results.md``, ``.json``, qualitative Grad-CAM PNG.
Reproduces: looped +4.0 pp over baseline on EuroSAT (frozen encoders).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jepa.data.roboflow_export import (
    build_eurosat_loaders,
    build_fgvc_aircraft_loaders,
    build_roboflow_dataloaders,
    download_roboflow_universe,
    download_zip_export,
    resolve_classification_dir,
)
from jepa.eval.linear_probe import (
    LinearProbeHead,
    _standardize_features,
    _train_probe_head,
    extract_features,
    load_encoder_from_checkpoint,
)
from jepa.eval.transfer_experiment import (
    encoder_gradcam,
    plot_qualitative_grid,
    run_frozen_encoder_transfer,
    save_results_json,
    train_scratch_classifier,
    write_results_markdown,
)
from jepa.utils.config import load_config
from jepa.utils.device import get_device
from jepa.utils.seed import set_seed

DEFENSE_PARAGRAPH = (
    "Aerial maritime perception is a core enabler for autonomous surface and airborne "
    "platforms: harbor monitoring, search-and-rescue, and coastal ISR all depend on "
    "robust object cues (vessels, docks, vehicles) under variable altitude and lighting. "
    "Self-supervised encoders pretrained on cheap unlabeled imagery can reduce labeled-data "
    "requirements for mission-specific fine-tuning, a practical constraint in defense and "
    "autonomy programs where expert annotation is scarce and deployment timelines are tight."
)

ROBOFLOW_DEFAULT = {
    "workspace": "demm",
    "project": "aerial-maritime-drone-dataset",
    "version": 1,
    "format": "yolov8",
}


def _loader_sizes(train_loader: DataLoader, val_loader: DataLoader) -> tuple[int, int]:
    return len(train_loader.dataset), len(val_loader.dataset)


def _prepare_dataset(args) -> tuple[DataLoader, DataLoader, dict]:
    img_size = args.img_size
    if args.source == "eurosat":
        return build_eurosat_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            img_size=img_size,
            max_train=args.max_train,
            max_val=args.max_val,
        )
    if args.source == "fgvc-aircraft":
        return build_fgvc_aircraft_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            img_size=img_size,
            max_train=args.max_train,
            max_val=args.max_val,
        )

    raw_dir = Path(args.data_dir)
    if args.download:
        if args.export_url:
            download_zip_export(args.export_url, raw_dir)
        else:
            download_roboflow_universe(
                args.workspace,
                args.project,
                args.version,
                raw_dir,
                model_format=args.roboflow_format,
            )
    cls_root = resolve_classification_dir(raw_dir)
    train_loader, val_loader, meta = build_roboflow_dataloaders(
        cls_root,
        batch_size=args.batch_size,
        img_size=img_size,
    )
    meta["name"] = f"Roboflow {args.workspace}/{args.project}"
    meta["dataset"] = "roboflow"
    return train_loader, val_loader, meta


def _qualitative_figures(
    args,
    train_loader: DataLoader,
    val_loader: DataLoader,
    meta: dict,
    out_dir: Path,
    device: torch.device,
) -> str:
    """Example predictions + Grad-CAM for the v3 baseline encoder."""
    baseline_cfg = load_config(args.baseline_config)
    model = load_encoder_from_checkpoint(baseline_cfg, args.baseline_checkpoint, device)
    num_classes = meta["num_classes"]
    embed_dim = baseline_cfg["encoder"]["embed_dim"]

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
        epochs=30,
        probe_lr=1e-3,
        weight_decay=1e-4,
        num_classes=num_classes,
        cosine_schedule=True,
    )

    images, labels = next(iter(val_loader))
    n = min(8, images.shape[0])
    images, labels = images[:n], labels[:n]
    head.eval()
    with torch.no_grad():
        feats = []
        for i in range(n):
            tokens = model.encoder.forward_all_patches(images[i : i + 1].to(device))
            feats.append(tokens.mean(dim=1).cpu())
        pooled = torch.cat(feats)
        pooled = (pooled - train_feats.mean(dim=0)) / train_feats.std(dim=0).clamp_min(1e-6)
        preds = head(pooled.to(device)).argmax(dim=1).cpu()

    grid_size = baseline_cfg["data"]["img_size"] // baseline_cfg["data"]["patch_size"]
    cams = []
    for i in range(n):
        cams.append(
            encoder_gradcam(
                model,
                images[i],
                head,
                grid_size,
                int(preds[i]),
                device,
            )
        )

    class_names = meta.get("class_names") or [str(i) for i in range(num_classes)]
    qual_path = out_dir / "qualitative_baseline_gradcam.png"
    plot_qualitative_grid(
        images,
        labels,
        preds,
        class_names,
        qual_path,
        title="Frozen v3 encoder: predictions + Grad-CAM (probe-guided)",
        cams=cams,
    )
    return f"Saved `{qual_path}`; green/red titles show correct vs incorrect predictions with probe-guided Grad-CAM overlays."


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboflow transfer learning experiment")
    parser.add_argument("--source", choices=("roboflow", "eurosat", "fgvc-aircraft"), default="roboflow")
    parser.add_argument("--data-dir", default="data/transfer/aerial_maritime")
    parser.add_argument("--download", action="store_true", help="Download Roboflow dataset before training")
    parser.add_argument("--export-url", default=None, help="Roboflow zip export URL (alternative to API key)")
    parser.add_argument("--workspace", default=ROBOFLOW_DEFAULT["workspace"])
    parser.add_argument("--project", default=ROBOFLOW_DEFAULT["project"])
    parser.add_argument("--version", type=int, default=ROBOFLOW_DEFAULT["version"])
    parser.add_argument("--roboflow-format", default=ROBOFLOW_DEFAULT["format"])
    parser.add_argument("--baseline-config", default="configs/image_jepa_cifar10_v3.yaml")
    parser.add_argument("--looped-config", default="configs/image_jepa_cifar10_v3_looped.yaml")
    parser.add_argument("--baseline-checkpoint", default="checkpoints/baseline_v3/latest.pt")
    parser.add_argument("--looped-checkpoint", default="checkpoints/baseline_v3_looped/latest.pt")
    parser.add_argument("--out-dir", default="results/transfer")
    parser.add_argument("--img-size", type=int, default=32, help="Resize (32 matches CIFAR JEPA pretraining)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--scratch-epochs", type=int, default=40)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="If Roboflow data missing, run FGVC Aircraft aerial proxy benchmark",
    )
    args = parser.parse_args()

    device = get_device("auto")
    set_seed(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_loader, val_loader, meta = _prepare_dataset(args)
    except Exception as exc:
        if args.source == "roboflow" and args.allow_fallback:
            print(f"Roboflow prep failed ({exc}); falling back to EuroSAT satellite proxy.")
            train_loader, val_loader, meta = build_eurosat_loaders(
                batch_size=args.batch_size,
                img_size=args.img_size,
                max_train=args.max_train,
                max_val=args.max_val,
            )
        else:
            raise

    train_n, val_n = _loader_sizes(train_loader, val_loader)
    meta["train_size"] = train_n
    meta["val_size"] = val_n
    print(f"Dataset: {meta.get('name', meta.get('dataset'))}  train={train_n}  val={val_n}  classes={meta['num_classes']}")

    baseline_cfg = load_config(args.baseline_config)
    looped_cfg = load_config(args.looped_config)
    rows: list[dict] = []
    t0 = time.time()

    print("\n=== Frozen v3 baseline encoder ===")
    base_res = run_frozen_encoder_transfer(
        baseline_cfg,
        args.baseline_checkpoint,
        train_loader,
        val_loader,
        device,
        label="frozen_v3_baseline",
        probe_epochs=args.probe_epochs,
    )
    base_res["notes"] = "Tuned linear probe, CIFAR-10 pretrained encoder"
    rows.append(base_res)
    print(f"  top1={base_res['top1_accuracy']:.2f}%  macro_f1={base_res['macro_f1']:.2f}%")

    print("\n=== Frozen looped predictor encoder ===")
    loop_res = run_frozen_encoder_transfer(
        looped_cfg,
        args.looped_checkpoint,
        train_loader,
        val_loader,
        device,
        label="frozen_v3_looped",
        probe_epochs=args.probe_epochs,
    )
    loop_res["notes"] = "Tuned linear probe, looped CIFAR-10 encoder"
    rows.append(loop_res)
    print(f"  top1={loop_res['top1_accuracy']:.2f}%  macro_f1={loop_res['macro_f1']:.2f}%")

    print("\n=== Scratch ResNet18 baseline ===")
    scratch_res = train_scratch_classifier(
        train_loader,
        val_loader,
        device,
        num_classes=meta["num_classes"],
        img_size=args.img_size,
        epochs=args.scratch_epochs,
    )
    scratch_res["notes"] = "ResNet18 trained from scratch on transfer data"
    rows.append(scratch_res)
    print(f"  top1={scratch_res['top1_accuracy']:.2f}%  macro_f1={scratch_res['macro_f1']:.2f}%")

    qual_note = _qualitative_figures(args, train_loader, val_loader, meta, out_dir, device)

    payload = {
        "dataset": meta,
        "results": rows,
        "elapsed_sec": time.time() - t0,
        "configs": {
            "baseline": args.baseline_config,
            "looped": args.looped_config,
        },
        "checkpoints": {
            "baseline": args.baseline_checkpoint,
            "looped": args.looped_checkpoint,
        },
    }
    save_results_json(out_dir / "transfer_results.json", payload)
    write_results_markdown(
        out_dir / "transfer_results.md",
        meta,
        rows,
        DEFENSE_PARAGRAPH,
        qual_note,
    )
    roboflow_note = """
## Running on Roboflow Aerial Maritime Drone (recommended)

```bash
export ROBOFLOW_API_KEY="..."   # free at https://app.roboflow.com/settings/api
python scripts/transfer_roboflow.py --download \\
  --workspace demm --project aerial-maritime-drone-dataset --version 1 \\
  --roboflow-format yolov8 --data-dir data/transfer/aerial_maritime
```

YOLO detection exports are auto-converted to image-level classification (dominant object class per image).
Alternatively, export **folder** format from the Roboflow UI and pass `--export-url <zip-link>`.
"""
    md_path = out_dir / "transfer_results.md"
    md_path.write_text(md_path.read_text() + roboflow_note)
    print(f"\nWrote {out_dir / 'transfer_results.md'}")
    print(f"Wrote {out_dir / 'transfer_results.json'}")


if __name__ == "__main__":
    main()
