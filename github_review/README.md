# GitHub publish review

*Review bundle for what will appear on GitHub. Generated for pre-push inspection.*

---

## What gets committed

### Documentation (portfolio layer)

| File | Role |
| --- | --- |
| [`README.md`](../README.md) | Project landing page |
| [`docs/IJEPA_Looped_Predictor_Report.md`](../docs/IJEPA_Looped_Predictor_Report.md) | **Main portfolio artifact** |
| [`portfolio_notes.md`](../portfolio_notes.md) | Interview / hiring talking points |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | How to extend |
| [`REPRODUCTION.md`](../REPRODUCTION.md) | Full reproduce guide |
| [`REPORT.md`](../REPORT.md) | v3 training report |
| [`results/README.md`](../results/README.md) | Results + figures index |
| [`model_cards/`](../model_cards/) | Professional model documentation |
| [`v3_baseline/`](../v3_baseline/) · [`v3_looped/`](../v3_looped/) | Version hubs |

### Code

| Path | Contents |
| --- | --- |
| `src/jepa/` | Core library (~15 modules) |
| `scripts/` | 18 CLI entry points + [`scripts/README.md`](../scripts/README.md) |
| `visualizations/` | Figure pipeline + rendered PNG/PDF |
| `configs/` | YAML configs + ablation variants |
| `app.py` + `demo/` | Gradio demo |
| `tests/` | pytest suite |

### Committed artifacts (results & figures)

```
results/
  ablations/summary.json, summary.md, *.md
  transfer/transfer_results.json, transfer_results.md, qualitative_*.png
visualizations/
  figures/*.png, *.pdf
  loop_analysis/*.png, *.pdf, summary.json
```

### Environment files

| File | Use |
| --- | --- |
| [`requirements.txt`](../requirements.txt) | pip / Hugging Face Spaces |
| [`requirements-dev.txt`](../requirements-dev.txt) | dev + pytest + roboflow |
| [`pyproject.toml`](../pyproject.toml) | uv / hatchling (preferred) |
| [`uv.lock`](../uv.lock) | Locked uv resolution |

---

## What stays local (gitignored)

```
.venv/
checkpoints/          # *.pt weights; train or distribute separately
data/                 # CIFAR-10 download
runs/                 # metrics.jsonl, training logs (except committed summaries)
logs/
__pycache__/, .pytest_cache/
```

**Important for reviewers:** A fresh clone runs tests and figures without checkpoints.
Headline probe numbers require `checkpoints/baseline_v3/latest.pt` (train ~5–6 h or supply weights).

---

## Fresh-clone verification checklist

Run from repository root after clone:

```bash
# 1. Environment
uv sync --extra dev
# or: python3.11 -m venv .venv && source .venv/bin/activate
#     pip install -r requirements-dev.txt

# 2. Verify install
python scripts/verify_install.py

# 3. Unit tests
python -m pytest tests/ -v

# 4. Fast figure smoke test (needs checkpoints if full pipeline)
python visualizations/generate_all_figures.py --fast

# 5. Demo (needs checkpoints for inference)
uv sync --extra demo && python app.py

# 6. Official probe (needs baseline checkpoint, ~15–20 min)
python scripts/linear_probe.py \
  --config configs/image_jepa_cifar10_v3.yaml \
  --checkpoint checkpoints/baseline_v3/latest.pt
```

Expected without checkpoints: steps 1–3 pass; 4–6 show clear missing-checkpoint messages or skip inference.

---

## Repository tree (committed)

```
looped-jepa/
├── README.md
├── CONTRIBUTING.md
├── portfolio_notes.md
├── REPRODUCTION.md
├── REPORT.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── uv.lock
├── app.py
├── v3_baseline/  v3_looped/
├── model_cards/
├── results/
├── docs/
│   └── IJEPA_Looped_Predictor_Report.md
├── src/jepa/
├── configs/
├── scripts/
├── visualizations/
├── demo/
└── tests/
```

---

## Headline numbers (for README consistency)

| Model | Tuned probe | Notes |
| --- | ---: | --- |
| v3 baseline | 77.23% | Reference |
| v3 looped | 75.13% | −2.10 pp |
| sandwich_rms | 78.28% | Best ablation |
| transfer looped | 76.75% | +4.0 pp vs baseline on EuroSAT |

---

## Pre-push checklist

- [ ] `python scripts/verify_install.py` passes
- [ ] `python -m pytest tests/ -q` passes
- [ ] README links resolve (report, model cards, results index)
- [ ] No secrets in `configs/`, `scripts/`, or env files
- [ ] Checkpoints distributed via LFS, release, or documented train command
- [ ] Gradio Space URL added to README when hosted
- [ ] Replace `<your-repo-url>` placeholders in README / report

---

## Suggested GitHub repository settings

- **Description:** Recurrent I-JEPA predictor on CIFAR-10: ablations, transfer, demo (~10M params)
- **Topics:** `jepa`, `self-supervised-learning`, `vision-transformer`, `world-models`, `pytorch`
- **Pin:** Technical report or README hero figure (`01_mask_reconstruction.png`)

---

*This folder is for local review only. Safe to commit; it documents the publish bundle.*
