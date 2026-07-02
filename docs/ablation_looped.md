# Looped Predictor Ablation (Smoke Run)

Date: 2026-06-29  
Config: `configs/ouro_smoke.yaml` (1 epoch, smaller encoder than v3)

## Results

| Loops | Linear probe top-1 | Latent loss | Train time (s) |
| --- | --- | --- | --- |
| 1 | 32.25% | 0.141 | 118 |
| 2 | 31.48% | 0.161 | 125 |

Checkpoints: `checkpoints/ablation_loops_{1,2}/latest.pt`  
Full JSON: `runs/ablation_results.json`

## Notes

- This is a **1-epoch smoke ablation** on the small `ouro_smoke` config (~3.3M params), not the full v3 baseline.
- 2 loops adds ~7% training time with slightly higher latent loss at this early stage.
- For publication-quality comparison, re-run at v3 scale:

```bash
python scripts/ablation_loops.py \
  --config configs/image_jepa_ouro_looped.yaml \
  --loops 1 2 4 \
  --epochs 50
```
