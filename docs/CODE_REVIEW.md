# Code Review: Phase 1 (2026-06-29)

## Bugs fixed

| Issue | Fix |
| --- | --- |
| `run_linear_probe` probed on augmented train images | Added `train_augment=False` |
| Gradio demo used a random probe head | Fits a probe on CIFAR-10 at startup via `train_probe_head` |
| `extract_features` left model in eval mode | Restores prior `training` flag |
| `compare_baseline.py` wrote to `/tmp` | Uses `tempfile.TemporaryDirectory` |
| Matplotlib figures not closed after save | `plt.close(fig)` in viz helpers |
| Duplicated checkpoint-loading logic | `load_encoder_from_checkpoint` in `linear_probe.py` |

## Structure

- Refactored `linear_probe.py`: shared `_standardize_features`, `_train_probe_head`, `train_probe_head`, `load_encoder_from_checkpoint`
- Scripts (`visualize`, `transfer_probe`, `ablation_loops`) use the shared loader

## Comment pass

- Trimmed marketing language ("official", "publishable ceiling", numbered forward steps)
- Shortened over-explained docstrings in `jepa.py`, `train.py`, `masking.py`, `vit.py`
- Kept the masking v2 collapse note; it's load-bearing context for anyone tuning masks

## Verified

```bash
python -m pytest tests/ -v
python scripts/linear_probe.py --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt
```
