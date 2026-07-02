# Predictor normalization (LayerNorm vs sandwich RMSNorm)

_Base config: `configs/ablations/base_v3_looped.yaml`_

| Variant | Tuned top-1 | feat_std | Mean loops (val) | Tail loss σ | Min feat_std (train) |
| --- | --- | --- | --- | --- | --- |
| layernorm | 75.36% | 0.1275 | 1.50 | 0.000 | 0.1275 |
| sandwich_rms | 78.28% | 0.0432 | 1.50 | 0.000 | 0.0432 |

## Loop usage detail

### layernorm

- Mean loops used: **1.500**
- Mean exit prob per loop: L1=0.500, L2=0.500
- Histogram (rounded loops): `{2: 10000}`

### sandwich_rms

- Mean loops used: **1.500**
- Mean exit prob per loop: L1=0.500, L2=0.500
- Histogram (rounded loops): `{2: 10000}`
