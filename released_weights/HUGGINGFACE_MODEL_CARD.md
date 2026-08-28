---
language: en
license: mit
library_name: looped-jepa
tags:
  - jepa
  - self-supervised
  - vision-transformer
  - cifar10
datasets:
  - cifar10
---

# looped-jepa — I-JEPA with a weight-shared looped predictor (~9.9M params)

**I-JEPA** on CIFAR-10 (32×32): ViT encoder (`384×5×6`) + predictor (`128×4×4`). The looped variant reuses one block stack for **2 refinement loops** with a learned **exit gate** (mean depth ≈ 1.5). Training: 300 epochs, RandAugment + tuned linear probe on frozen features.

## Weight files (registry keys)

| File | Tuned top-1 | feat_std |
| --- | ---: | ---: |
| `baseline_v3/latest.pt` | **77.23%** | 0.1607 |
| `looped_v3/latest.pt` | 75.13% (−2.10 pp) | 0.1450 |
| `sandwich_rmsnorm/latest.pt` | **78.28%** | 0.0432 |

Load: `from jepa import load_ijepa; model = load_ijepa("baseline_v3", pretrained=True)`

## Related results

- **EuroSAT transfer** (frozen encoders): baseline **72.75%** → looped **76.75%** (+4.0 pp).
- Default looped LayerNorm **trails** in-domain CIFAR; **sandwich-RMSNorm** beats baseline (+1.05 pp).
- Exit gate is ~uniform on CIFAR; adaptive depth is architectural, not yet learned.

**License:** MIT · **Code:** [github.com/JMangold0352/looped-jepa](https://github.com/JMangold0352/looped-jepa) · Weights: URLs in repo `released_weights/urls.yaml` (not on PyPI until published).
