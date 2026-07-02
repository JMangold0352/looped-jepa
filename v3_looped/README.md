# v3 Looped Predictor

Version hub for the **I-JEPA CIFAR-10 v3 looped predictor**: a weight-shared recurrent
predictor with a learned exit gate. Same encoder + recipe as the baseline.

> This folder is the landing page for the looped version. Shared code, configs, and
> weights live in the common tree and are referenced below (checkpoints and data are
> gitignored). See the full [model card](../model_cards/v3_looped.md).

## Headline

| Tuned probe | vs baseline | Aerial transfer | Best ablation |
| ---: | ---: | ---: | ---: |
| **75.13%** | −2.10 pp | **+4 pp** (76.75%) | **78.28%** (sandwich-RMSNorm) |

## Artifacts

| Artifact | Path |
| --- | --- |
| Model card | [`model_cards/v3_looped.md`](../model_cards/v3_looped.md) |
| Config | [`configs/image_jepa_cifar10_v3_looped.yaml`](../configs/image_jepa_cifar10_v3_looped.yaml) |
| Checkpoint | `checkpoints/baseline_v3_looped/latest.pt` |
| Training metrics | `runs/cifar10_v3_looped/metrics.jsonl` |
| Ablation results | [`results/ablations/`](../results/ablations/) |
| Per-loop deep dive | [`visualizations/loop_analysis/`](../visualizations/loop_analysis/) |
| Baseline comparison | `runs/looped_v3_comparison.json` |

## Train

```bash
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3_looped.yaml
```

## Compare against the baseline

```bash
python scripts/compare_looped_v3.py \
  --baseline-checkpoint checkpoints/baseline_v3/latest.pt \
  --looped-checkpoint checkpoints/baseline_v3_looped/latest.pt
```

## Per-loop analysis

```bash
python visualizations/generate_all_figures.py --loop-analysis-only
```

## Load in Python

```python
import torch
from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config

cfg = load_config("configs/image_jepa_cifar10_v3_looped.yaml")
model = IJEPA.from_config(cfg)
ckpt = torch.load("checkpoints/baseline_v3_looped/latest.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
```
