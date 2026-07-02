# Visualization Suite

Publication ready figures comparing the V3 baseline

## Quick start

```bash
cd ~/Projects/looped-jepa
source .venv/bin/activate

# Optional: t-SNE embeddings (falls back to PCA if missing)
pip install scikit-learn

# Full figure set (~30–90 min on MPS depending on hardware)
python visualizations/generate_all_figures.py

# Smoke test (~2 min)
python visualizations/generate_all_figures.py --fast
```

## Overnight run

```bash
bash scripts/run_visualizations_overnight.sh
```

Logs: `runs/visualizations_generate.log`

## Outputs

All figures are written to `visualizations/figures/` as **PNG + PDF** at **300 DPI**:

| File stem | Description |
|-----------|-------------|
| `01_mask_reconstruction` | Original / masked context / embedding-space prediction quality (baseline vs looped) |
| `02_attention_maps` | Predictor attention to context patches (1 pass vs per-loop) |
| `03_embeddings` | 2D embedding projection (t-SNE or PCA), baseline vs looped encoders |
| `03_per_loop_cosine` | Cosine similarity vs teacher targets across predictor loops |
| `04_training_curves` | Loss, probe accuracy, feat_std over epochs |
| `04_expected_loops_training` | Expected loop depth during training |
| `04_exit_loop_distribution` | Validation exit-loop histogram + CDF |
| `05_ablation_summary` | Bar charts for all ablation variants |

**Step 6** writes per-loop deep-dive figures to `visualizations/loop_analysis/` (see [loop_analysis/README.md](loop_analysis/README.md)). Use `--loop-analysis-only` or `--skip-loop-analysis` to control it.

## Custom paths

```bash
python visualizations/generate_all_figures.py \
  --baseline-checkpoint checkpoints/baseline_v3/latest.pt \
  --looped-checkpoint checkpoints/baseline_v3_looped/latest.pt \
  --out-dir visualizations/figures \
  --embed-method tsne
```

## Adding a new figure

1. Add a plotting function under `visualizations/figures/`.
2. Import and call it from `visualizations/generate_all_figures.py`.
3. Use `visualizations.style.save_figure()` for consistent PNG/PDF export.

## Per-loop analysis

Deeper per-sample loop diagnostics live in `visualizations/loop_analysis/`:

```bash
python visualizations/generate_all_figures.py --loop-analysis-only
python visualizations/generate_all_figures.py --loop-analysis-only --fast
```

See [loop_analysis/README.md](loop_analysis/README.md) for output descriptions.
