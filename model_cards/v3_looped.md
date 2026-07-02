# Model Card: I-JEPA CIFAR-10 **v3 Looped Predictor**

A **weight-shared recurrent predictor** variant of the v3 baseline (Ouroboros-style, meaning
the predictor feeds its own refined output back as input). The encoder, augmentation, and
training schedule are identical to v3; the only change is that the standard ViT predictor is
wrapped in a `LoopedPredictor` that reuses the same block stack for multiple refinement steps,
with a learned per-loop **exit gate**.

| Field | Value |
| --- | --- |
| Task | Self-supervised representation learning (image) |
| Method | I-JEPA + looped (recurrent, weight-shared) predictor |
| Dataset | CIFAR-10, 50k train / 10k val, 32×32, 10 classes |
| Encoder | Identical to v3 baseline (ViT `384×5×6`) |
| Predictor | `LoopedPredictor`, `max_loops=2`, exit gate on |
| Trainable params | ≈ 9.87M (predictor is weight-shared across loops) |
| Precision / device | fp32, trained on Apple MPS |
| Config | [`configs/image_jepa_cifar10_v3_looped.yaml`](../configs/image_jepa_cifar10_v3_looped.yaml) |
| Checkpoint | `checkpoints/baseline_v3_looped/latest.pt` |
| Version hub | [`v3_looped/`](../v3_looped/) |

---

## 1. Architecture summary

Same I-JEPA setup as the [v3 baseline](v3_baseline.md), with a recurrent predictor:

```
context latents ─►┌───────────────────────────────┐
                  │  LoopedPredictor              │   loop t = 1 … max_loops
                  │  ┌─────────────────────────┐  │
                  │  │ shared ViT block stack  │◄─┼── refined state feeds back
                  │  └───────────┬─────────────┘  │
                  │        exit gate p_t  ─────────┼─► P(stop at loop t)
                  └───────────────┴───────────────┘
                                  │
                     predicted target latents ─► smooth-L1 vs EMA teacher
```

- **Weight sharing**: the predictor block stack is reused for each loop, so extra depth
  costs no extra parameters, a compute/quality knob at fixed capacity.
- **Exit gate**: produces per-loop exit probabilities; an **entropy regularizer**
  (`exit_entropy_beta = 0.01`) discourages degenerate always-early / always-late exits.
- **This checkpoint**: `max_loops=2`, LayerNorm predictor, exit gate enabled. The mean
  exit depth on the validation set is ≈ 1.5 loops.

Key code:

- `src/jepa/models/looped_predictor.py`
- `src/jepa/models/predictor.py` (`ouro_ready`, `forward_stack`)
- `src/jepa/eval/loop_metrics.py`

## 2. Training details

Identical recipe to the v3 baseline (300 epochs, bs 256, AdamW `2e-3`, wd `0.05`,
15-epoch warmup → cosine, RandAugment(2,9) + RRC). Looped-specific settings:

| Hyperparameter | Value |
| --- | --- |
| `predictor.looped` | true |
| `predictor.max_loops` | 2 |
| `predictor.use_exit_gate` | true |
| `predictor.norm` | layer |
| `exit_entropy_beta` | 0.01 |

```bash
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3_looped.yaml
```

## 3. Performance

**Official metric: tuned linear probe** on frozen features (same protocol as baseline).

| Metric | v3 looped | v3 baseline | Δ |
| --- | ---: | ---: | ---: |
| Val top-1 (tuned probe) | **75.13%** | 77.23% | −2.10 pp |
| Best probe LR | 3e-3 | 3e-3 | n/a |
| `feat_std` | 0.1450 | 0.1607 | n/a |

Probe LR sweep (looped): `3e-4 → 73.22%`, `1e-3 → 74.79%`, `3e-3 → 75.13%`.
Source: `runs/looped_v3_comparison.json`.

### Where looping helps

On in-domain CIFAR-10 linear probing the default 2-loop LayerNorm variant trails the
baseline. Two findings qualify this:

1. **Normalization matters more than loop count.** With a **sandwich-RMSNorm** predictor
   the looped model reaches **78.28%**, the best result in the ablation suite, above the
   77.23% baseline. See [`results/ablations/`](../results/ablations/).
2. **Looping helps *transfer*.** On the aerial transfer benchmark the frozen looped
   encoder beats the frozen baseline by **+4 pp** top-1 (76.75% vs 72.75%). See
   [`transfer.md`](transfer.md).

### Per-loop behavior

Deep-dive figures (exit distribution, per-loop cosine/L1, difficulty vs loops, early/late
exit examples) are generated to [`visualizations/loop_analysis/`](../visualizations/loop_analysis/):

```bash
python visualizations/generate_all_figures.py --loop-analysis-only
```

## 4. Limitations

- The **default** (2-loop, LayerNorm) checkpoint underperforms the baseline on in-domain
  linear probe; the win requires sandwich-RMSNorm or a transfer setting.
- More loops (4) did **not** monotonically help and lowered `feat_std`.
- The exit gate adds hyperparameters (`max_loops`, `exit_entropy_beta`) that need tuning.
- Same 32×32, single-view, linear-probe-only caveats as the baseline.

## 5. Intended use

- Research on **adaptive-compute** / recurrent predictors for self-supervised learning.
- A drop-in variant to study depth-vs-parameters trade-offs at fixed capacity.
- Preferred over the baseline when **transfer** to out-of-domain imagery is the goal.

### Defense & autonomy relevance

Adaptive-depth inference is attractive for autonomous platforms with **variable compute
budgets**: the exit gate lets the model spend more refinement loops on hard inputs and
fewer on easy ones, trading latency for quality per sample rather than globally. Combined
with the observed transfer advantage on aerial imagery, the looped encoder is a strong
candidate when adapting a compact pretrained backbone to mission-specific intelligence,
surveillance, and reconnaissance (ISR) tasks under edge constraints.

## 6. How to load & run

```python
import torch
from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config

cfg = load_config("configs/image_jepa_cifar10_v3_looped.yaml")
model = IJEPA.from_config(cfg)
ckpt = torch.load("checkpoints/baseline_v3_looped/latest.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

# Frozen patch features (encoder is identical in shape to the baseline):
features = model.encoder.forward_all_patches(images)  # (B, 64, 384)
```

Compare directly against the baseline:

```bash
python scripts/compare_looped_v3.py \
  --baseline-checkpoint checkpoints/baseline_v3/latest.pt \
  --looped-checkpoint checkpoints/baseline_v3_looped/latest.pt
```

## 7. Citation

```bibtex
@misc{mangold2025jepav3looped,
  title  = {I-JEPA CIFAR-10 v3 Looped Predictor},
  author = {John Mangold},
  year   = {2025},
  note   = {Weight-shared recurrent predictor with learned exit gate}
}
```
