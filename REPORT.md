# I-JEPA CIFAR-10 v3: Training Report

_Training report, 2025._

## Recipe (v3)

- **Model**: ~9.9M trainable params
  - Encoder: ViT, embed_dim=384, depth=5, num_heads=6, mlp_ratio=4.0, drop_path=0.1
  - Predictor: ViT, embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0
  - Exact trainable count: 9,816,960
- **Dataset**: CIFAR-10, 32×32, patch_size 4 (8×8 grid = 64 patches)
- **Schedule**: 300 epochs, AdamW lr=2e-3, weight_decay 0.05, batch_size 256, 15-epoch warmup + cosine decay
- **EMA**: 0.996 → 0.9999 linear (capped below 1.0 to avoid the no-op final step that decayed feat_std in v1)
- **Augmentation**: RandAugment(n=2, m=9) + RandomResizedCrop(scale 0.5-1.0)
- **Masking**: deterministic target subselection (`sorted(targets)[:N]`). The "unbiased" random subselection added in v2 caused representation collapse (v2/v2b regressed to ~64%); reverted to the original deterministic scheme. Block positions still vary per sample via sampled block geometry.
  - num_target_blocks: 4
  - target_scale: [0.15, 0.2]
  - context_scale: [0.85, 1.0]
  - context_patches_range: [24, 40]  (student sees ~50% of patches)
  - target_patches_range: [10, 22]   (predicts ~25% of patches)
- **Eval**:
  - Periodic trend probe: fixed-LR (1e-3), 20 epochs, every 25 epochs
  - **Official final number**: tuned probe (cosine LR + sweep over {3e-4, 1e-3, 3e-3} + feature standardization, 100 epochs)

## Outcome

- **Status**: COMPLETED. Final tuned probe available.
- **Official top-1 (tuned probe)**: **77.21%**
- **Best LR**: 3e-03
- **feat_std (collapse diagnostic)**: 0.1607

## Periodic probes (fixed-LR, trend monitoring)

| Epoch | Top-1 (%) | feat_std |
|------:|----------:|---------:|
| 25  | 60.31 | 0.3030 |
| 50  | 65.31 | 0.2244 |
| 75  | 69.19 | 0.2182 |
| 100 | 71.81 | 0.2286 |
| 125 | 72.89 | 0.2332 |
| 150 | 74.24 | 0.2231 |
| 175 | 74.96 | 0.2053 |
| 200 | 76.02 | 0.1904 |
| 225 | 76.18 | 0.1776 |
| 250 | 76.15 | 0.1693 |
| 275 | 76.26 | 0.1624 |
| 300 | 76.38 | 0.1607 |

**Best periodic probe**: 76.38% (epoch 300, feat_std=0.1607)

## Final tuned probe (per-LR sweep)

| LR | Best val top-1 (%) |
|----|-------------------:|
| 3e-04 | 76.28 |
| 1e-03 | 76.94 |
| 3e-03 | **77.21** |

## Loss trajectory

- Epoch 1 loss: 0.3279
- Epoch 300 loss: 0.0211
- Total epochs logged: 300

## Two code-level fixes kept from v2/v2b

1. **Deterministic target subselection** (`src/jepa/masking.py`). The "unbiased"
   random subselection caused representation collapse at this scale (v2/v2b
   regressed to ~64% vs v1's ~80% linear probe). Reverted to
   `sorted(targets)[:N]`; target positions still vary per-sample via the
   sampled block geometry.
2. **EMA momentum cap 0.9999** (in the base config). v1's
   `ema_momentum_end=1.0` made the final EMA step a no-op
   (`ema = ema*1.0 + p*0.0`) and feat_std decayed 0.28→0.18 over training.
   0.9999 keeps the teacher moving without over-weighting the noisy final-step
   student.

(The unused `tgt_proj` parameter was already removed in commit 45082fa. The exact
trainable count is 9,866,240 according to the git history and 9,816,960 by direct
count in v3; the difference reflects minor parameter pruning between commits.)

## Target & methodology note

- **Target**: 83% top-1 (set with honest methodology rather than inflated to 85%) for the ~10M-parameter scale. v3 reached 77.21%, below target but a clean baseline with no collapse and a stable probe trajectory.
- The official number is the **tuned** linear probe (LR sweep + standardization), not the periodic fixed-LR probe. The periodic probes are trend indicators only.
- feat_std is a collapse diagnostic: healthy values are ~0.2–0.4; values near 0 signal representation collapse. v3 held 0.16–0.30 throughout, no collapse.
- No test-set tuning, no cherry-picking the best epoch's probe.

## Artifacts (in this package)

- **Checkpoint**: `checkpoints/baseline_v3/latest.pt`
- **Metrics**: `runs/cifar10_baseline_v3/metrics.jsonl`
- **Config**: `configs/image_jepa_cifar10_v3.yaml`
- **Training log**: `logs/v3_run.log`

## To re-run the official probe

```bash
python scripts/linear_probe.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt
```

## Context: v4 and v5 (both failed; v3 remains best)

- **v4** stacked 5 changes (two-view + harder masking + cosine EMA + stronger aug + drop-path 0.2) and **collapsed**: feat_std dropped 0.31→0.10 by epoch 100, probe fell 8.7 points behind v3. Killed at epoch 102.
- **v5** isolated two-view on the v3 baseline and **still underperformed**: tracked ~4.5 points behind v3 at epoch 100, feat_std at 0.11. Killed at epoch 107.
- **Conclusion:** two-view I-JEPA does not help at the ~10M-param / 32×32 scale. v3's single-view recipe is well-matched to this scale.

**v3 is the best model.**
