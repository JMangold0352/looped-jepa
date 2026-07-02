# Scripts reference

All commands assume the **repository root** as the working directory and an activated
virtualenv (`source .venv/bin/activate`).

Quick verify after clone: `python scripts/verify_install.py`

---

## Training & evaluation

### `train_jepa.py`

Train I-JEPA from a YAML config. Auto-resumes from `checkpoints/<run>/latest.pt` unless
`--no-auto-resume`.

```bash
# v3 baseline (300 epochs, ~5–6 h on MPS)
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3.yaml

# v3 looped predictor
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3_looped.yaml

# Fresh start (ignore existing checkpoint)
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3.yaml --no-auto-resume
```

**Outputs:** `checkpoints/<checkpoint_dir>/latest.pt`, `runs/<run_dir>/metrics.jsonl`  
**Reproduces:** headline checkpoints when trained from scratch.

---

### `linear_probe.py`

Official evaluation: tuned linear probe on a frozen encoder (LR sweep + standardization).

```bash
python scripts/linear_probe.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--tuned` | on | LR sweep; use `--no-tuned` for fixed-LR trend probe |
| `--epochs` | 100 | Probe training epochs |

**Expected:** ~77.23% top-1, feat_std ≈ 0.16 (baseline). ~15–20 min on MPS.

---

### `compare_looped_v3.py`

Head-to-head tuned probe: v3 baseline vs default looped checkpoint.

```bash
python scripts/compare_looped_v3.py
python scripts/compare_looped_v3.py --out runs/looped_v3_comparison.json
```

**Outputs:** JSON with per-model probe results and delta (expected **−2.10 pp**).

---

### `compare_baseline.py`

Compare any two checkpoints (or vs random init).

```bash
python scripts/compare_baseline.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --baseline checkpoints/baseline_v3/latest.pt \
  --candidate checkpoints/baseline_v3_looped/latest.pt \
  --tuned
```

---

### `reprobe_tuned.py`

Re-run tuned probe with `num_workers=0` (useful in sandboxes or when dataloader workers fail).

```bash
python scripts/reprobe_tuned.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt
```

---

## Ablations

### `run_ablations.py`

Train and/or evaluate the three ablation suites (loop count, exit entropy, sandwich norm).

```bash
# Full suite: train all variants (300 epochs each)
python scripts/run_ablations.py --suite all --train

# Evaluate only (checkpoints must exist)
python scripts/run_ablations.py --suite all --eval-only

# Smoke test (1 epoch)
python scripts/run_ablations.py --suite loop_count --train --epochs 1
```

**Outputs:** `results/ablations/summary.json`, `summary.md`, per-suite JSON/MD.

---

### `ablation_loops.py`

Legacy quick loop-depth sweep (shorter runs than the full ablation suite).

```bash
python scripts/ablation_loops.py \
  --config configs/ouro_smoke.yaml \
  --loops 1 2 4 \
  --epochs 2
```

**Outputs:** `runs/ablation_results.json`

---

### `monitor_ablations.py`

Poll ablation training log; finalize reports when complete.

```bash
python scripts/monitor_ablations.py
python scripts/monitor_ablations.py --finalize-if-done
```

Reads `runs/ablations_train.log`; writes Desktop review files when training finishes.

---

## Transfer learning

### `transfer_roboflow.py`

Frozen-encoder transfer: baseline vs looped vs scratch ResNet18.

```bash
# EuroSAT aerial proxy (no API key)
python scripts/transfer_roboflow.py --source eurosat

# Roboflow maritime (requires ROBOFLOW_API_KEY)
export ROBOFLOW_API_KEY="..."
python scripts/transfer_roboflow.py --download \
  --workspace demm --project aerial-maritime-drone-dataset --version 1 \
  --roboflow-format yolov8 --data-dir data/transfer/aerial_maritime
```

**Outputs:** `results/transfer/transfer_results.md`, `.json`, qualitative PNG.

---

### `transfer_probe.py`

Linear probe on CIFAR-100 or a folder dataset.

```bash
python scripts/transfer_probe.py \
  --dataset cifar100 \
  --checkpoint checkpoints/baseline_v3/latest.pt \
  --out runs/transfer_cifar100.json
```

---

### `download_roboflow.py`

Download a Roboflow export into `train/<class>/` + `val/<class>/` layout.

```bash
export ROBOFLOW_EXPORT_URL="https://..."
python scripts/download_roboflow.py --out-dir data/transfer
```

---

## Visualizations

### `visualize.py`

Legacy single-checkpoint plots (metrics, mask overlay, PCA, probe sweep).

```bash
python scripts/visualize.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt \
  --out-dir runs/visualizations
```

**Prefer:** `python visualizations/generate_all_figures.py` for the full publication suite.

---

### `generate_looped_performance_review.py`

Write a Desktop markdown review after looped training completes.

```bash
python scripts/generate_looped_performance_review.py
```

---

## Demo

### `gradio_demo.py` (deprecated)

Forwards to `app.py`. Use the root entry point instead:

```bash
uv sync --extra demo && python app.py
```

---

## Install verification

### `verify_install.py`

Smoke-check imports, configs, and optional checkpoints after a fresh clone.

```bash
python scripts/verify_install.py
python scripts/verify_install.py --require-checkpoints   # fail if weights missing
```
