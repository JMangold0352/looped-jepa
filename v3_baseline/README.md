# v3 Baseline

Version hub for the **I-JEPA CIFAR-10 v3 baseline**: the publication reference encoder.

> This folder is the landing page for the baseline version. Shared code, configs, and
> weights live in the common tree and are referenced below (checkpoints and data are
> gitignored). See the full [model card](../model_cards/v3_baseline.md).

## Headline

| Tuned linear probe | `feat_std` | Params |
| ---: | ---: | ---: |
| **77.23%** | 0.1607 | ~9.9M |

## Artifacts

| Artifact | Path |
| --- | --- |
| Model card | [`model_cards/v3_baseline.md`](../model_cards/v3_baseline.md) |
| Config | [`configs/image_jepa_cifar10_v3.yaml`](../configs/image_jepa_cifar10_v3.yaml) |
| Checkpoint | `checkpoints/baseline_v3/latest.pt` |
| Training metrics | `runs/cifar10_baseline_v3/metrics.jsonl` |
| Ablation results | [`results/ablations/`](../results/ablations/) |
| Figures | [`visualizations/figures/`](../visualizations/figures/) |

## Train

```bash
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3.yaml
```

## Evaluate (official metric)

```bash
python scripts/linear_probe.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt
```

## Load in Python

```python
import torch
from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config

cfg = load_config("configs/image_jepa_cifar10_v3.yaml")
model = IJEPA.from_config(cfg)
ckpt = torch.load("checkpoints/baseline_v3/latest.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
```
