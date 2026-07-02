# Loop count (1 vs 2 vs 4)

_Base config: `configs/ablations/base_v3_looped.yaml`_

| Variant | Tuned top-1 | feat_std | Mean loops (val) | Tail loss σ | Min feat_std (train) |
| --- | --- | --- | --- | --- | --- |
| loops_1 | 77.24% | 0.1609 | 1.00 | 0.000 | 0.1609 |
| loops_2 | 75.04% | 0.1276 | 1.50 | n/a | n/a |
| loops_4 | 75.49% | 0.1049 | 1.88 | 0.000 | 0.1049 |

## Loop usage detail

### loops_1

- Mean loops used: **1.000**
- Histogram (rounded loops): `{1: 10000}`

### loops_2

- Mean loops used: **1.500**
- Mean exit prob per loop: L1=0.500, L2=0.500
- Histogram (rounded loops): `{2: 10000}`

### loops_4

- Mean loops used: **1.875**
- Mean exit prob per loop: L1=0.500, L2=0.500, L3=0.500, L4=0.500
- Histogram (rounded loops): `{2: 10000}`
