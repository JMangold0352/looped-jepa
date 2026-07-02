# Model Card: I-JEPA CIFAR-10 **v3 Baseline**

The publication baseline: a self-supervised Vision Transformer encoder trained with
Image-JEPA (I-JEPA) on CIFAR-10 under a strict ~10M-parameter budget.

| Field | Value |
| --- | --- |
| Task | Self-supervised representation learning (image) |
| Method | I-JEPA (masked latent prediction, EMA teacher) |
| Dataset | CIFAR-10, 50k train / 10k val, 32×32, 10 classes |
| Encoder | ViT, `embed_dim=384`, `depth=5`, `heads=6`, patch 4×4 → 64 tokens |
| Predictor | Narrow ViT, `embed_dim=128`, `depth=4`, `heads=4` (non-looped) |
| Trainable params | **9.87M** (encoder ≈ 8.9M + predictor ≈ 1.0M) |
| Precision / device | fp32, trained on Apple MPS |
| Config | [`configs/image_jepa_cifar10_v3.yaml`](../configs/image_jepa_cifar10_v3.yaml) |
| Checkpoint | `checkpoints/baseline_v3/latest.pt` |
| Version hub | [`v3_baseline/`](../v3_baseline/) |

---

## 1. Architecture summary

I-JEPA learns representations by predicting the **latent** embeddings of masked target
regions from a visible context region, with no pixel reconstruction and no negatives.

```
image (3×32×32)
  └─ patchify 4×4 ──────────────► 64 tokens (8×8 grid), 384-dim
        │
   ┌────┴─────┐
   │ student  │  ViT encoder (depth 5, 6 heads)  ── context tokens ─┐
   │ encoder  │                                                     │
   └──────────┘                                          ┌──────────▼─────────┐
   ┌──────────┐                                          │ ViT predictor      │
   │ teacher  │  EMA copy of student ── target tokens ──►│ (depth 4, 4 heads) │
   │ encoder  │  (stop-grad)                             │ predicts latents   │
   └──────────┘                                          └──────────┬─────────┘
                                                                    │
                        smooth-L1 in latent space ◄─────────────────┘
```

- **EMA teacher**: momentum ramped `0.996 → 0.9999` (linear). The cap below `1.0` keeps
  the teacher tracking the student on the final steps; at `1.0` the last update is a
  no-op and `feat_std` collapses.
- **Masking**: multi-block, context 24–40 patches, target 10–22 patches, with
  **deterministic** target subselection (`sorted(targets)[:N]`). Random subselection
  caused representation collapse at this scale (documented in the config header).
- **Head geometry**: 384 / 6 = 64-dim encoder heads; 128 / 4 = 32-dim predictor heads.

## 2. Training details

| Hyperparameter | Value |
| --- | --- |
| Epochs | 300 |
| Batch size | 256 |
| Optimizer | AdamW, lr `2e-3`, weight decay `0.05` |
| Schedule | 15-epoch warmup → cosine decay |
| Grad clip | 1.0 |
| Stochastic depth | drop-path 0.1 (linear across encoder blocks) |
| Augmentation | RandAugment(n=2, m=9) + conservative RRC (scale 0.5–1.0) |
| Loss | smooth-L1 in latent space |

```bash
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3.yaml
```

## 3. Performance

**Official metric: tuned linear probe** (cosine LR schedule, LR sweep
`{3e-4, 1e-3, 3e-3}`, feature standardization) on frozen features.

| Metric | Value |
| --- | ---: |
| Val top-1 (tuned probe) | **77.23%** |
| Best probe LR | 3e-3 |
| `feat_std` | 0.1607 |

Probe LR sweep: `3e-4 → 76.23%`, `1e-3 → 76.97%`, `3e-3 → 77.23%`.

### Predictor ablations (v3 recipe, 300 epochs each)

The baseline uses a standard non-looped predictor. Looped variants and normalization
choices were ablated; the **sandwich-RMSNorm** predictor is the strongest configuration:

| Variant | Tuned top-1 | `feat_std` | Mean loops |
| --- | ---: | ---: | ---: |
| loops_1 (≈ baseline predictor) | 77.24% | 0.1609 | 1.00 |
| loops_2 | 75.04% | 0.1276 | 1.50 |
| loops_4 | 75.49% | 0.1049 | 1.88 |
| entropy_off | 76.00% | 0.1270 | 1.55 |
| entropy_on | 75.36% | 0.1275 | 1.50 |
| layernorm | 75.36% | 0.1275 | 1.50 |
| **sandwich_rms** | **78.28%** | 0.0432 | 1.50 |

Full tables + JSON: [`results/ablations/`](../results/ablations/).

```bash
python scripts/linear_probe.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt
```

## 4. Limitations

- Trained and evaluated at **32×32**; transfer to higher resolutions requires resizing
  and is approximate (see [`transfer.md`](transfer.md)).
- **Single-view** I-JEPA; two-view variants (v2/v2b) regressed at this scale.
- Linear-probe evaluation only; no end-to-end fine-tuning results are claimed here.
- Small-scale academic dataset; absolute accuracy is not directly comparable to
  ImageNet-scale encoders.

## 5. Intended use

- Research and education on self-supervised vision encoders under tight parameter budgets.
- A **frozen feature extractor** for linear probing and lightweight transfer.
- The reference baseline for the looped-predictor and transfer experiments in this repo.

### Defense & autonomy relevance

Label-efficient perception matters most where annotation is scarce and deployment
timelines are tight, exactly the regime of many defense and autonomy programs. A
compact (<10M param) self-supervised encoder can be pretrained on abundant *unlabeled*
sensor imagery and then adapted to mission-specific tasks (e.g. aerial/maritime ISR,
harbor monitoring, search-and-rescue) with only a small labeled head. The transfer
experiments in this repo demonstrate the frozen encoder generalizing to aerial imagery;
the small footprint is compatible with edge/embedded inference on autonomous platforms.

## 6. How to load & run

```python
import torch
from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config

cfg = load_config("configs/image_jepa_cifar10_v3.yaml")
model = IJEPA.from_config(cfg)
ckpt = torch.load("checkpoints/baseline_v3/latest.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

# Frozen patch features for downstream probing:
# images: (B, 3, 32, 32) normalized with CIFAR-10 mean/std
features = model.encoder.forward_all_patches(images)  # (B, 64, 384)
```

## 7. Citation

```bibtex
@misc{jepa_cifar10_v3,
  title  = {I-JEPA CIFAR-10 v3 Baseline},
  author = {jepa-ouro},
  year   = {2026},
  note   = {Self-supervised ViT encoder under 10M parameters}
}
```
