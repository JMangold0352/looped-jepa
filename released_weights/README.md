# Released pretrained weights

Three CIFAR-10 I-JEPA checkpoints are documented here. **Weights are not stored in git**
(see `.gitignore`). Download with [`scripts/download_weights.sh`](../scripts/download_weights.sh)
or load via Python (see below).

All models share the v3 training recipe unless noted: ViT encoder (`embed_dim=384`, `depth=5`,
`heads=6`), predictor (`embed_dim=128`, `depth=4`, `heads=4`), **9,816,960 trainable parameters**
(~9.9M). Official metric: tuned linear probe on frozen features (300-epoch pretrain).

| Registry key | Config | Checkpoint path | Tuned top-1 | `feat_std` | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `baseline_v3` | [`configs/image_jepa_cifar10_v3.yaml`](../configs/image_jepa_cifar10_v3.yaml) | `checkpoints/baseline_v3/latest.pt` | **77.23%** | 0.1607 | Non-looped reference |
| `looped_v3` | [`configs/image_jepa_cifar10_v3_looped.yaml`](../configs/image_jepa_cifar10_v3_looped.yaml) | `checkpoints/baseline_v3_looped/latest.pt` | 75.13% | 0.1450 | 2-loop + exit gate, LayerNorm (−2.10 pp vs baseline) |
| `sandwich_rmsnorm` | [`configs/image_jepa_cifar10_v3_looped_sandwich_rms.yaml`](../configs/image_jepa_cifar10_v3_looped_sandwich_rms.yaml) | `checkpoints/ablations/sandwich_norm/sandwich_rms/latest.pt` | **78.28%** | 0.0432 | Best ablation; sandwich RMSNorm + loops |

**License:** [MIT](../LICENSE) (same as the repository).

## Download URLs

URLs live in [`released_weights/urls.yaml`](urls.yaml). Replace `PLACEHOLDER_URL` with real
Hugging Face or Google Drive links before publishing weights.

| Registry key | Hugging Face | Google Drive |
| --- | --- | --- |
| `baseline_v3` | TODO | TODO |
| `looped_v3` | TODO | TODO |
| `sandwich_rmsnorm` | TODO | TODO |

### Environment overrides

Set any of these to override the YAML map (useful in CI or mirrors):

- `LOOPED_JEPA_WEIGHT_URL_BASELINE_V3`
- `LOOPED_JEPA_WEIGHT_URL_LOOPED_V3`
- `LOOPED_JEPA_WEIGHT_URL_SANDWICH_RMSNORM`

Optional fallback-only overrides (used when Hugging Face is unset or fails):

- `LOOPED_JEPA_WEIGHT_GDRIVE_BASELINE_V3`
- `LOOPED_JEPA_WEIGHT_GDRIVE_LOOPED_V3`
- `LOOPED_JEPA_WEIGHT_GDRIVE_SANDWICH_RMSNORM`

## Shell download

```bash
# List registry entries and whether checkpoints are already present
./scripts/download_weights.sh --list

# Download all released weights (skips files that already exist)
./scripts/download_weights.sh

# Download one variant
./scripts/download_weights.sh baseline_v3
```

## Python loader

```python
from jepa import load_ijepa

model = load_ijepa("baseline_v3", pretrained=True, device="cpu")
# also: "looped_v3", "sandwich_rmsnorm"
```

If `pretrained=True` and the checkpoint is missing locally, the loader attempts a download
when URLs are configured. With placeholder URLs, use `scripts/download_weights.sh` after
URLs are published, or pass `checkpoint="/path/to/latest.pt"`.

## Quick sanity check (64 validation images)

```bash
uv sync --extra dev
./scripts/download_weights.sh baseline_v3   # once URLs are live
python scripts/quickstart_forward.py --model baseline_v3
```

Prints mean feature-norm standard deviation (`feat_std`) on 64 CIFAR-10 validation images.
