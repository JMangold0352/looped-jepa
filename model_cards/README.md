# Model Cards

Professional model cards for every released model in this repository. Each card documents
architecture, training, performance (linear probe + ablations), limitations, intended use
(including defense/autonomy relevance), and load-and-run instructions.

| Card | Summary | Headline metric |
| --- | --- | ---: |
| [**v3_baseline.md**](v3_baseline.md) | I-JEPA ViT encoder, non-looped predictor (~9.9M params) | 77.23% tuned probe |
| [**v3_looped.md**](v3_looped.md) | Weight-shared recurrent predictor + exit gate | 75.13% probe · +4 pp transfer |
| [**transfer.md**](transfer.md) | Frozen-encoder downstream probing (aerial, CIFAR-100) | 76.75% aerial (looped) |

## At a glance

| Model | Config | Checkpoint | Version hub |
| --- | --- | --- | --- |
| v3 baseline | `configs/image_jepa_cifar10_v3.yaml` | `checkpoints/baseline_v3/latest.pt` | [`v3_baseline/`](../v3_baseline/) |
| v3 looped | `configs/image_jepa_cifar10_v3_looped.yaml` | `checkpoints/baseline_v3_looped/latest.pt` | [`v3_looped/`](../v3_looped/) |

The two versions share the same encoder architecture and training recipe; they are
separated **by config + checkpoint** (and mirrored by the `v3_baseline/` and `v3_looped/`
hub folders) rather than by duplicating shared code.
