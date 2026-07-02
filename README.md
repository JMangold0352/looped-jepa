<div align="center">

# Recurrent Latent Prediction with I-JEPA on CIFAR-10

**A compact, reproducible Image-JEPA stack with a weight-shared looped predictor, built for interpretable world modeling under a 10M-parameter budget.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch&logoColor=white)](pyproject.toml)
[![Gradio demo](https://img.shields.io/badge/demo-Gradio-f97316?logo=gradio&logoColor=white)](app.py)
[![Params ~10M](https://img.shields.io/badge/params-~9.9M-54A24B)](model_cards/v3_baseline.md)
[![License: research](https://img.shields.io/badge/license-research%20%2F%20educational-lightgrey)](#license)

*Self-supervised ViT encoders · masked latent prediction · adaptive-depth recurrence · publication figures · transfer to aerial imagery*

[Quickstart](#installation--quickstart) ·
[Results](#key-results) ·
[Gallery](#visual-gallery) ·
[Report](docs/IJEPA_Looped_Predictor_Report.md) ·
[Demo](#gradio-demo) ·
[Model cards](model_cards/) ·
[Reproduce](REPRODUCTION.md)

</div>

---

## Hook

**Image-JEPA** (LeCun et al.) learns visual representations by predicting *latent* embeddings of masked regions from context, with no pixels and no contrastive negatives. This repository implements a **faithful, end-to-end I-JEPA pipeline** on CIFAR-10 and asks a question central to embodied AI and world models:

> *What if the predictor that fills in missing latent structure is **recurrent**, refining its guess over multiple shared-weight steps, with a learned exit gate that spends compute only where needed?*

This project releases everything needed to reproduce that study: two trained models (baseline and looped), a **seven-variant ablation suite**, publication-quality visualizations (including how attention evolves across loops), aerial transfer experiments, and an interactive Gradio demo. The entire stack stays under **~9.9M trainable parameters**, small enough for edge deployment and fast iteration.

---

## Motivation & why a looped predictor?

JEPA-style training treats perception as **predictive coding in representation space**: the encoder sees a partial view, and the predictor infers what the full scene *means* in latent space, supervised by an exponential moving average (EMA) teacher. That is already a primitive world model, but the standard predictor runs as a **single feed-forward pass**.

**Recurrent latent dynamics** mirror how agents refine beliefs step by step: early loops capture coarse structure, and later loops resolve ambiguity. Because the loop reuses one **weight-shared** block stack, this added depth costs **zero extra parameters**; only the amount of computation grows. A learned **exit gate** then makes that depth adaptive, so easy inputs stop early while hard inputs keep refining.

Two terms recur throughout this README:

- **Exit gate** — a small learned head that, after each loop, estimates the probability that refinement can stop. It turns a fixed-depth predictor into an adaptive-depth one.
- **Sandwich RMSNorm** — RMS normalization applied both *before and after* each attention and feed-forward sub-layer in the predictor (rather than only once, before each sub-layer). This paired placement stabilizes the shared-weight loop and, as the ablations show, is what makes recurrence actually pay off.

This repository isolates the predictor change while holding the v3 encoder recipe fixed, so the comparisons stay honest. Along the way I document stability lessons that matter at small scale:

| Design choice | Rationale |
| --- | --- |
| Deterministic target subselection | Random subselection caused representation collapse (~64% probe); reverted to `sorted(targets)[:N]` |
| EMA momentum cap at 0.9999 | At 1.0 the final EMA step is a no-op; `feat_std` decayed in v1 |
| Exit-gate entropy regularization | Prevents degenerate always-early / always-late exits |
| Sandwich RMSNorm in the predictor | Strongest ablation (+1.05 pp over baseline); normalization > raw loop count |
| RandAugment + mild random-resized crop (RRC) at 32×32 | Strong augmentation without destroying 4×4 patch structure |

The default looped checkpoint **does not beat** the baseline on in-domain CIFAR-10 probing (−2.1 pp). That negative result is informative: recurrence alone is insufficient without the right predictor normalization, and the **transfer** and **ablation** stories are where the science lives.

---

## Key results

Official metric: **tuned linear probe** on frozen features (cosine learning-rate schedule, sweep over `{3e-4, 1e-3, 3e-3}`, feature standardization, 300-epoch pretraining). Accuracy differences are reported in percentage points (pp).

### Released models

| Model | Tuned top-1 | `feat_std` | Params | Notes |
| --- | ---: | ---: | ---: | --- |
| **[v3 baseline](v3_baseline/)** | **77.23%** | 0.1607 | 9.87M | Publication reference encoder |
| [v3 looped](v3_looped/) | 75.13% | 0.1450 | 9.87M | 2-loop + exit gate; **−2.10 pp** vs baseline |
| [sandwich-RMSNorm](results/ablations/) | **78.28%** | 0.0432 | 9.87M | Best ablation; looped + sandwich norm |

### Full ablation suite (300 epochs each, v3 recipe)

| Variant | Tuned top-1 | `feat_std` | Mean loops |
| --- | ---: | ---: | ---: |
| loops_1 | 77.24% | 0.1609 | 1.00 |
| loops_2 | 75.04% | 0.1276 | 1.50 |
| loops_4 | 75.49% | 0.1049 | 1.88 |
| entropy_on | 75.36% | 0.1275 | 1.50 |
| entropy_off | 76.00% | 0.1270 | 1.55 |
| layernorm | 75.36% | 0.1275 | 1.50 |
| **sandwich_rms** | **78.28%** | 0.0432 | 1.50 |

**Takeaways:**

- Default looped predictor (LayerNorm, 2 loops): **−2.1 pp** in-domain; recurrence without the right norm hurts `feat_std` and probe accuracy.
- **Normalization dominates loop count:** sandwich-RMSNorm beats both baseline and all other ablations.
- **Transfer flips the story:** frozen looped encoder **+4.0 pp** over frozen baseline on aerial imagery (see below).
- Per-loop analysis: mean cosine gain loop 1 → final ≈ **+0.21**; exit gate ≈ **50% / 50%** at loops 1 and 2 (expected depth **1.5**).

Details: [`results/ablations/summary.md`](results/ablations/summary.md)

---

### Defense and Autonomy Applications
The looped predictor’s adaptive compute design is well-suited for resource-constrained edge systems common in defense and autonomous platforms. 

The learned exit gate allows the model to allocate additional computation only when needed, which is valuable for deployment on drones and other unmanned systems. Strong transfer performance to an aerial maritime drone dataset further supports its potential for real-world autonomy applications.

---

## Visual gallery

All figures are generated at **300 DPI** (PNG + PDF). Regenerate with [`visualizations/generate_all_figures.py`](visualizations/generate_all_figures.py).

<table>
<tr>
<td width="50%">

**Masked latent prediction: baseline vs looped**

Target patches tinted by cosine similarity to the EMA teacher (greener = better).

<img src="visualizations/figures/01_mask_reconstruction.png" width="100%" alt="Mask reconstruction comparison"/>

</td>
<td width="50%">

**Predictor attention across loops**

Where the looped predictor looks in context; attention sharpens with refinement.

<img src="visualizations/figures/02_attention_maps.png" width="100%" alt="Attention maps"/>

</td>
</tr>
<tr>
<td>

**Embedding space (t-SNE)**

Frozen encoder features: baseline vs looped.

<img src="visualizations/figures/03_embeddings.png" width="100%" alt="Embedding comparison"/>

</td>
<td>

**Per-loop cosine to teacher**

Aggregate refinement curve across validation batches.

<img src="visualizations/figures/03_per_loop_cosine.png" width="100%" alt="Per-loop cosine"/>

</td>
</tr>
<tr>
<td>

**Exit-loop distribution**

Learned adaptive depth on the validation set.

<img src="visualizations/figures/04_exit_loop_distribution.png" width="100%" alt="Exit loop distribution"/>

</td>
<td>

**Ablation summary**

All seven predictor variants, tuned probe.

<img src="visualizations/figures/05_ablation_summary.png" width="100%" alt="Ablation summary"/>

</td>
</tr>
<tr>
<td colspan="2">

**Per-loop deep dive** ([`visualizations/loop_analysis/`](visualizations/loop_analysis/)): exit stats, cosine/L1 by loop, difficulty vs loops, early/late exit examples.

<img src="visualizations/loop_analysis/03_loops_vs_difficulty.png" width="100%" alt="Loops vs difficulty"/>

</td>
</tr>
</table>

More: [`visualizations/README.md`](visualizations/README.md)

---

## Installation & quickstart

```bash
git clone https://github.com/JMangold0352/looped-jepa.git && cd looped-jepa
uv sync --extra dev          # or: pip install -r requirements-dev.txt
source .venv/bin/activate

python scripts/verify_install.py
python -m pytest tests/test_shapes.py -v

# Official evaluation: requires checkpoint (train or copy weights)
python scripts/linear_probe.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt
```

**Optional extras**

| Extra | Install | Use |
| --- | --- | --- |
| `demo` | `uv sync --extra demo` | Gradio app (`app.py`) |
| `viz` | `uv sync --extra viz` | t-SNE embeddings in figure suite |
| `transfer` | `uv sync --extra transfer` | Roboflow / EuroSAT transfer |

**Load encoder in Python**

```python
import torch
from jepa.models.jepa import IJEPA
from jepa.utils.config import load_config

cfg = load_config("configs/image_jepa_cifar10_v3.yaml")
model = IJEPA.from_config(cfg)
ckpt = torch.load("checkpoints/baseline_v3/latest.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
features = model.encoder.forward_all_patches(images)  # (B, 64, 384)
```

---

## Reproduce training, evaluation & visualizations

### Train

```bash
# v3 baseline (~300 epochs, MPS/CUDA)
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3.yaml

# v3 looped predictor (same recipe, recurrent predictor + exit gate)
python scripts/train_jepa.py --config configs/image_jepa_cifar10_v3_looped.yaml
```

### Evaluate

```bash
# Tuned linear probe (official metric)
python scripts/linear_probe.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt

# Head-to-head baseline vs looped
python scripts/compare_looped_v3.py \
  --baseline-checkpoint checkpoints/baseline_v3/latest.pt \
  --looped-checkpoint checkpoints/baseline_v3_looped/latest.pt

# Full ablation suite (train + eval all 7 variants)
python scripts/run_ablations.py --suite all --train
```

### Visualizations

```bash
# Full publication figure set + per-loop deep dive (~30–90 min)
python visualizations/generate_all_figures.py

# Smoke test (~2 min)
python visualizations/generate_all_figures.py --fast

# Per-loop analysis figures only
python visualizations/generate_all_figures.py --loop-analysis-only
```

Full reproduce-from-scratch guide: [**REPRODUCTION.md**](REPRODUCTION.md) · Experiment write-up: [**REPORT.md**](REPORT.md)

---

## Gradio demo

Interactive side-by-side comparison: upload any image, toggle **1 / 2 / 4** predictor loops, watch **attention evolve loop-by-loop**, inspect exit-gate stats, and optionally run a **CIFAR-10 linear probe** on frozen features.

```bash
uv sync --extra demo && python app.py    # http://127.0.0.1:7860
```

| | |
| --- | --- |
| **Local** | [`app.py`](app.py) · [`demo/README.md`](demo/README.md) |
| **Hugging Face Spaces** | *Coming soon: `app.py` is the Space entry point; see `requirements.txt`* |

> The shipped looped checkpoint was trained with `max_loops=2`. Selecting **4 loops** in the demo extrapolates beyond training (the UI shows a caveat).

---

## Transfer learning

Frozen-encoder transfer (backbone not fine-tuned; linear probe on top). Primary benchmark: **EuroSAT RGB** as an aerial/satellite proxy (1500 train / 400 val). Roboflow *Aerial Maritime Drone* runs with `--download` when `ROBOFLOW_API_KEY` is set.

| Method | Top-1 | Macro F1 | Notes |
| --- | ---: | ---: | --- |
| frozen v3 baseline | 72.75% | 75.66% | CIFAR-10 pretrained |
| **frozen v3 looped** | **76.75%** | 75.43% | **+4.0 pp** vs baseline |
| scratch ResNet18 | 77.50% | 67.06% | Trained on transfer data only |

```bash
python scripts/transfer_roboflow.py --source eurosat

# Roboflow Aerial Maritime Drone dataset (requires an API key)
export ROBOFLOW_API_KEY="..."
python scripts/transfer_roboflow.py --download \
  --workspace demm --project aerial-maritime-drone-dataset --version 1 \
  --roboflow-format yolov8 --data-dir data/transfer/aerial_maritime
```

Qualitative saliency: `results/transfer/qualitative_baseline_gradcam.png` · Full write-up: [`results/transfer/transfer_results.md`](results/transfer/transfer_results.md) · [**Transfer model card**](model_cards/transfer.md)

**CIFAR-100** (label-space shift): 46.32% top-1 with frozen v3 baseline (vs 77.2% in-domain).

---

## Model cards

Professional cards with architecture diagrams, training recipes, performance tables, limitations, defense/autonomy relevance, and load-and-run snippets.

| Card | Summary |
| --- | --- |
| [**v3 baseline**](model_cards/v3_baseline.md) | I-JEPA ViT encoder, non-looped predictor, **77.23%** |
| [**v3 looped**](model_cards/v3_looped.md) | Weight-shared recurrence + exit gate, 75.13% probe, +4 pp transfer |
| [**transfer**](model_cards/transfer.md) | Frozen-encoder downstream probing |
| [**Index**](model_cards/README.md) | All cards + version hubs |

Version hubs: [`v3_baseline/`](v3_baseline/) · [`v3_looped/`](v3_looped/)

---

## Relevance to defense, autonomy & edge AI

The core constraints in this project (compact models, label efficiency, and interpretable inference) align directly with the requirements of autonomous and defense perception systems.

| Theme | Connection |
| --- | --- |
| **Aerial & maritime world models** | Encoders pretrained on abundant unlabeled imagery transfer to aerial domains; the looped variant improves transfer by +4 pp |
| **Planning & latent dynamics** | Predicting scene *structure* in latent space is a building block for model-based reinforcement learning and predictive world models |
| **Adaptive compute** | The exit gate allocates depth per sample, spending more computation on difficult inputs where latency budgets allow |
| **Interpretability** | Per-loop attention maps, exit-depth distributions, and mask-reconstruction panels make predictor behavior directly inspectable |
| **Edge deployment** | A sub-10M-parameter, 32×32-native stack is compatible with embedded inference after mission-specific adaptation |
| **Label efficiency** | A frozen self-supervised backbone with a small linear head reduces annotation requirements for new classes |

This is research code rather than a deployed system, but the end-to-end workflow (train → ablate → visualize → transfer → demo) is intended to be readable and straightforward to extend for autonomy and defense research.

---

## Repository layout

```
looped-jepa/
├── v3_baseline/ · v3_looped/     Version hubs (config + checkpoint pointers)
├── model_cards/                  Professional model documentation
├── results/                      Ablation + transfer JSON/MD summaries
├── visualizations/               Figure code + rendered outputs
├── app.py · demo/                Gradio interactive demo
├── src/jepa/                     Core library (models, train, eval, masking)
├── configs/ · scripts/ · tests/
├── checkpoints/ · data/ · runs/  (gitignored, local artifacts)
└── docs/                         Technical report, code review
```

---

## Citation

If you use this codebase or checkpoints in your work, please cite:

```bibtex
@misc{mangold2025loopedjepa,
  title        = {Recurrent Latent Prediction with I-JEPA on CIFAR-10},
  author       = {John Mangold},
  year         = {2025},
  howpublished = {\url{https://github.com/JMangold0352/looped-jepa}},
  note         = {Self-supervised ViT encoders with a looped predictor under 10M parameters}
}
```

**Acknowledgments**

- [**I-JEPA**](https://arxiv.org/abs/2301.08243): LeCun, Assran, et al.; masked latent prediction framework
- [**Vision Transformer**](https://arxiv.org/abs/2010.11929): Dosovitskiy et al.
- **Ouroboros / recurrent predictor** lineage: weight-shared depth as a compute knob at fixed capacity
- **CIFAR-10**: Krizhevsky; **EuroSAT**: aerial transfer proxy; **Roboflow**: maritime drone dataset API

---

## Documentation index

| Doc | Contents |
| --- | --- |
| [**IJEPA Looped Predictor Report**](docs/IJEPA_Looped_Predictor_Report.md) | Full technical report and extended write-up |
| [**results/README.md**](results/README.md) | Results, figures, and report links |
| [**CONTRIBUTING.md**](CONTRIBUTING.md) | How to extend the codebase |
| [**scripts/README.md**](scripts/README.md) | Every CLI entry point |
| [REPRODUCTION.md](REPRODUCTION.md) | Reproduce training and evaluation from scratch |
| [REPORT.md](REPORT.md) | v3 training report and v4/v5 negative results |
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | Synthesis report |
| [visualizations/README.md](visualizations/README.md) | Figure pipeline |

---

## License

Research and educational use. See config headers for experiment lineage and version history.
