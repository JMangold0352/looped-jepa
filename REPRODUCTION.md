# Reproduction Guide: I-JEPA CIFAR-10 v3

_Reproduction guide · updated for the fresh-clone workflow._

This guide walks through reproducing the v3 result (77.23% tuned linear probe) from a
**fresh clone**. It assumes macOS with Apple Silicon (MPS) or a CUDA GPU. CPU works but
is ~10× slower.

## Fresh clone (start here)

```bash
git clone https://github.com/JMangold0352/looped-jepa.git && cd looped-jepa

# Environment (pick one)
uv sync --extra dev
# OR: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt

source .venv/bin/activate
python scripts/verify_install.py
python -m pytest tests/test_shapes.py -v
```

**Note:** `checkpoints/` and `data/` are gitignored. The verify script passes without
weights. For headline probe numbers you need `checkpoints/baseline_v3/latest.pt`, either
train (Step 4) or copy pretrained weights into place.

All paths below are **relative to the repository root**.

## Prerequisites

- Python 3.11
- PyTorch 2.x with MPS or CUDA support
- torchvision, tqdm, pyyaml

## Step 1: Set up the environment

```bash
cd looped-jepa

# Option A: plain venv (simplest)
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision tqdm pyyaml

# Option B: uv (faster, uses the included uv.lock)
uv sync
source .venv/bin/activate
```

## Step 2: Verify the installation

Run the unit tests to confirm the code is intact:

```bash
python -m pytest tests/test_shapes.py -v
```

You should see 6 tests pass. These test forward shapes, the mask collator,
looped predictor, VICReg regularizer, and config inheritance.

## Step 3: Run the official probe on the included checkpoint

The package includes the trained v3 checkpoint. Run the official tuned linear
probe to confirm the headline number:

```bash
python scripts/linear_probe.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt
```

Expected output:
```
Tuned linear probe results:
  Best LR: 3.00e-03
  Best val top-1: 77.21%
  feat_std: 0.1607
Per-LR results: ...
```

(This takes ~15-20 minutes on MPS: 3 LRs × 100 epochs each.)

## Step 4: Reproduce training from scratch (optional, ~5-6 hours)

To retrain v3 from scratch:

```bash
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3.yaml
```

This will:
1. Download CIFAR-10 to `data/` (already included in this package)
2. Build the ~9.9M-param I-JEPA model
3. Train 300 epochs with the v3 recipe
4. Run a fixed-LR probe every 25 epochs (trend monitoring)
5. Run the tuned linear probe at the end (official number)
6. Save checkpoints to `checkpoints/baseline_v3/latest.pt`
7. Log metrics to `runs/cifar10_baseline_v3/metrics.jsonl`

**Expected timing on Apple Silicon (MPS):** ~5-6 hours total
- ~4 hours pure training (~57 sec/epoch × 300 epochs)
- ~50 minutes periodic probes (12 × 20-epoch probes)
- ~25 minutes final tuned probe (3 LRs × 100 epochs)

**Output you should see:**
```
Using device: mps
Trainable parameters: 9,816,960
epoch 1/300  loss=0.3279
  [probe] epoch 25  top1=60.31%  feat_std=0.3030
  [probe] epoch 50  top1=65.31%  feat_std=0.2244
...
  [probe] epoch 300  top1=76.38%  feat_std=0.1607
  [tuned-probe] running final tuned linear probe (LR sweep)...
  [tuned-probe] lr=3e-04  best_val_top1=76.28%
  [tuned-probe] lr=1e-03  best_val_top1=76.94%
  [tuned-probe] lr=3e-03  best_val_top1=77.21%
  [tuned-probe] FINAL top1=77.21%  best_lr=3e-03  feat_std=0.1607
Training complete. Checkpoint: checkpoints/baseline_v3/latest.pt
```

## Step 5: Verify reproducibility

The training script seeds with `cfg["seed"] = 42` (set in
`configs/image_jepa_cifar10.yaml`). With the same seed, the same config, and
the same PyTorch version, you should reproduce the 77.21% within ~0.2%
(random variation from MPS nondeterminism).

## Config inheritance

The v3 config inherits from the base config:

```
configs/image_jepa_cifar10_v3.yaml
  └── _base_: image_jepa_cifar10.yaml  ← base (v1) settings
```

The base config defines the model architecture, masking geometry, base
training schedule, and the EMA cap (0.9999). The v3 config overrides:
- `augmentation`: RandAugment(n=2, m=9) + RRC(0.5-1.0)
- `train.epochs`: 300 (up from 200)
- `train.run_dir` / `checkpoint_dir`: v3-specific paths
- `eval`: tuned probe settings (cosine LR + sweep + standardization)

To see the fully-resolved config (base + overrides merged), run:

```python
from jepa.utils.config import load_config
cfg = load_config("configs/image_jepa_cifar10_v3.yaml")
import json; print(json.dumps(cfg, indent=2))
```

## Key files to review

1. **`src/jepa/models/jepa.py`**: core I-JEPA: encoder + EMA target encoder + predictor, the `forward()` method, `update_target_encoder()`.
2. **`src/jepa/masking.py`**: the I-JEPA mask collator. Read the comment block explaining why deterministic subselection is used (the v2 collapse).
3. **`src/jepa/train.py`**: training loop, EMA schedule (`ema_schedule`), probe integration (`run_probe`, `run_probe_tuned`).
4. **`src/jepa/data/cifar10.py`**: augmentation pipeline (`build_transforms`), supports `default` / `randaugment` / `aggressive_color_jitter` kinds.
5. **`src/jepa/eval/linear_probe.py`**: both the simple fixed-LR probe and the tuned LR-sweep probe.
6. **`configs/image_jepa_cifar10_v3.yaml`**: the exact v3 config with inline rationale for each setting.

## Troubleshooting

**`RuntimeError: torch_shm_manager: Operation not permitted`**
This happens when PyTorch's DataLoader workers can't access shared memory. It's
a sandbox restriction in some environments. Fix: run outside any sandbox, or
set `data.num_workers: 0` in the config (slower but works everywhere).

**`ModuleNotFoundError: No module named 'torchvision'`**
Install dependencies: `pip install torch torchvision tqdm pyyaml`.

**Probe accuracy is 0% after only 2 epochs**
This is expected; the probe needs ~20 epochs to converge. The smoke test
configs (`v4_smoke.yaml`, `v5_smoke.yaml`) only run 2 epochs to verify the
training loop works, not to get a meaningful accuracy.

**Training is killed silently after ~50 minutes**
This was an issue with macOS / Cursor background-process reaping. Fix: run
training inside a `screen` session:
```bash
screen -dmS train bash -c 'python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3.yaml > logs/v3_run.log 2>&1'
screen -r train  # to attach and watch
```
