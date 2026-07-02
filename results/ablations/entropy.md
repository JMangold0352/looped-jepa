# Exit-gate entropy regularization (on vs off)

_Base config: `configs/ablations/base_v3_looped.yaml`_

| Variant | Tuned top-1 | feat_std | Mean loops (val) | Tail loss σ | Min feat_std (train) |
| --- | --- | --- | --- | --- | --- |
| entropy_on | 75.36% | 0.1275 | 1.50 | 0.000 | 0.1275 |
| entropy_off | 76.00% | 0.1270 | 1.55 | 0.000 | 0.1270 |

## Loop usage detail

### entropy_on

- Mean loops used: **1.500**
- Mean exit prob per loop: L1=0.500, L2=0.500
- Histogram (rounded loops): `{2: 10000}`

### entropy_off

- Mean loops used: **1.547**
- Mean exit prob per loop: L1=0.453, L2=0.454
- Histogram (rounded loops): `{2: 9998, 1: 2}`
