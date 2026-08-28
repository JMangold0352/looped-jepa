# Scale check: does the CIFAR looped penalty hold at higher class count / resolution?

**Hypothesis (one sentence):** If the CIFAR penalty shrinks or flips at 200 classes / 64px, recurrence is scale-sensitive; if it holds, the penalty is robust and the EuroSAT gain is a domain effect.

This folder tracks a **≤3 GPU-day** comparison using the **v3 recipe** (same encoder/predictor widths, tuned linear probe protocol). TinyImageNet training is **not started here** — configs and probe script are ready; fill the TBD row after a 300-epoch run (single seed 42 is acceptable if labeled).

## Results table

| dataset | classes | baseline top-1 | looped top-1 | Δ (pp) | feat_std baseline | feat_std looped | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CIFAR-10 | 10 | 77.23% | 75.13% | −2.10 | 0.1607 | 0.1450 | published |
| EuroSAT (transfer) | 10 | 72.75% | 76.75% | +4.00 | 0.1389 | 0.1023 | published |
| TinyImageNet-200 | 200 | TBD | TBD | TBD | TBD | TBD | **TBD** |

Machine-readable source: [`scale_table.json`](scale_table.json).

## Architecture note (TinyImageNet)

| Setting | CIFAR-10 v3 | TinyImageNet v3 |
| --- | --- | --- |
| Resolution | 32×32 | **64×64** |
| Patch size | 4 | **8** |
| Token grid | 8×8 (64 patches) | 8×8 (64 patches) |
| Classes | 10 | **200** |
| Encoder | 384-d, depth 5 | **unchanged** |
| Predictor | 128-d, depth 4 | **unchanged** |
| Trainable params | ~9.82M | **~9.87M** (+55k from 8×8 patch embed vs 4×4) |

Only `data.dataset`, `img_size`, `patch_size`, and `num_classes` change in config; masking geometry is unchanged.

## Reproduce

```bash
# Train (after you are ready — not run in the scaffold PR)
python scripts/train_jepa.py --config configs/image_jepa_tinyimagenet_v3.yaml
python scripts/train_jepa.py --config configs/image_jepa_tinyimagenet_v3_looped.yaml

# Probe (same protocol as CIFAR)
python scripts/scale_probe.py \
  --config configs/image_jepa_tinyimagenet_v3.yaml \
  --checkpoint checkpoints/tinyimagenet_baseline_v3/latest.pt \
  --out results/scale/runs.json

# Smoke (1 epoch, verifies loader + train loop)
python scripts/train_jepa.py --config configs/image_jepa_tinyimagenet_smoke.yaml
```

## Figure

After TinyImageNet numbers exist, regenerate the bar chart:

```bash
python visualizations/figures/scale_bars.py
# writes visualizations/figures/06_scale_comparison.png (+ .pdf)
```

Existing CIFAR/EuroSAT rows already render; TinyImageNet bars are omitted until `baseline_top1` is set in `scale_table.json`.
