# Looped Predictor Ablation Summary

All runs use the v3 training recipe (300 epochs, RandAugment, tuned linear probe).

## Loop count (1 vs 2 vs 4)

| Variant | Tuned top-1 | feat_std | Mean loops |
| --- | --- | --- | --- |
| loops_1 | 77.24% | 0.1609 | 1.00 |
| loops_2 | 75.04% | 0.1276 | 1.50 |
| loops_4 | 75.49% | 0.1049 | 1.88 |

## Exit-gate entropy regularization (on vs off)

| Variant | Tuned top-1 | feat_std | Mean loops |
| --- | --- | --- | --- |
| entropy_on | 75.36% | 0.1275 | 1.50 |
| entropy_off | 76.00% | 0.1270 | 1.55 |

## Predictor normalization (LayerNorm vs sandwich RMSNorm)

| Variant | Tuned top-1 | feat_std | Mean loops |
| --- | --- | --- | --- |
| layernorm | 75.36% | 0.1275 | 1.50 |
| sandwich_rms | 78.28% | 0.0432 | 1.50 |
