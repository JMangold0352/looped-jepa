# Per-loop deep dive (Prompt 3)

Per-sample analysis of the **looped v3 predictor** (`checkpoints/baseline_v3_looped/latest.pt`). These figures complement the aggregate plots in `visualizations/figures/` (steps 03 and 04).

## Outputs

| File | Description |
|------|-------------|
| `01_exit_distribution` | Histogram + CDF of exit loop; mean P(exit) per loop index |
| `02_cosine_l1_by_loop` | Mean cosine similarity and smooth-L1 vs loop, with error bars |
| `02_cosine_l1_by_loop_per_class` | Per-class Δcosine (loop 1 → final loop) |
| `03_loops_vs_difficulty` | Scatter: loop-1 cosine vs expected exit loops |
| `04_early_vs_late_examples` | Example images: early exit vs late exit (same class) |
| `summary.json` | Numeric aggregates (exit stats, per-loop metrics) |

All figures are saved as PNG + PDF at 300 DPI.

## Generate

From the repo root:

```bash
# Full pipeline (step 6 of generate_all_figures.py)
python visualizations/generate_all_figures.py

# Loop analysis only
python visualizations/generate_all_figures.py --loop-analysis-only

# Smoke test (fewer batches)
python visualizations/generate_all_figures.py --loop-analysis-only --fast

# Skip loop analysis when running all figures
python visualizations/generate_all_figures.py --skip-loop-analysis
```

## Notes

- Uses the default looped checkpoint unless you pass `--looped-checkpoint`.
- `04_loops_vs_difficulty` treats low loop-1 cosine as higher difficulty (more refinement needed).
- Early/late exit panels pick representative samples from the validation set with diverse classes.
