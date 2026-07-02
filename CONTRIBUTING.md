# Contributing & extending

Thank you for exploring this codebase. This is a research repository, not a production
library. Contributions should stay focused, reproducible, and well-documented.

## Before you start

```bash
git clone https://github.com/JMangold0352/looped-jepa.git && cd looped-jepa
uv sync --extra dev          # or: pip install -r requirements-dev.txt
source .venv/bin/activate
python scripts/verify_install.py
python -m pytest tests/ -v
```

All paths in configs and scripts are **relative to the repository root**. Run commands from
the root directory.

## Project layout (where to edit)

| Area | Path | Purpose |
| --- | --- | --- |
| Core library | `src/jepa/` | Models, training loop, masking, eval, utils |
| Configs | `configs/` | YAML experiment definitions (`_base_` inheritance) |
| Entry points | `scripts/` | CLI tools (see [`scripts/README.md`](scripts/README.md)) |
| Figures | `visualizations/` | Publication figure pipeline |
| Demo | `app.py`, `demo/` | Gradio interactive demo |
| Results | `results/` | Committed summaries (JSON + MD); large artifacts gitignored |
| Tests | `tests/` | Shape, ablation, visualization smoke tests |

Checkpoints (`checkpoints/`), run logs (`runs/`), and datasets (`data/`) are **gitignored**.
Reproduce them locally via training or the steps in [`REPRODUCTION.md`](REPRODUCTION.md).

## How to extend

### Add a new predictor variant

1. Implement or configure in `src/jepa/models/predictor.py` or `looped_predictor.py`.
2. Add flags under `predictor:` in a new YAML under `configs/`.
3. Register an ablation variant in `src/jepa/ablations/registry.py` if it belongs in the suite.
4. Run `python scripts/run_ablations.py --suite <name> --train --epochs 1` as a smoke test.
5. Update `results/ablations/` summaries via `--eval-only` after full training.

### Add a new dataset or transfer benchmark

1. Add a loader in `src/jepa/data/` (follow `roboflow_export.py` or `cifar10.py`).
2. Wire it into `scripts/transfer_roboflow.py` or `scripts/transfer_probe.py`.
3. Write results to `results/transfer/` using `src/jepa/eval/transfer_experiment.py` helpers.

### Add a new figure

1. Add a plotting function under `visualizations/figures/`.
2. Import and call it from `visualizations/generate_all_figures.py`.
3. Use `visualizations.style.save_figure()` for consistent PNG/PDF export.
4. Add a smoke assertion in `tests/test_generate_figures.py` if appropriate.

### Add a new config / training recipe

1. Create `configs/my_experiment.yaml` with `_base_: image_jepa_cifar10_v3.yaml`.
2. Override only what changes (`train.run_dir`, `predictor`, etc.).
3. Train: `python scripts/train_jepa.py --config configs/my_experiment.yaml`.
4. Document expected metrics in a model card under `model_cards/`.

## Code style

- Match existing patterns: type hints, `from __future__ import annotations`, minimal scope.
- Prefer extending existing functions over parallel implementations.
- Docstrings on public modules and CLI entry points; explain *why* for non-obvious training choices.
- No secrets in commits (API keys, `.env`, credentials).

## Testing

```bash
python -m pytest tests/test_shapes.py -v           # fast sanity
python -m pytest tests/test_ablations.py -v        # ablation registry
python visualizations/generate_all_figures.py --fast
```

## Reporting results

When an experiment produces headline numbers:

1. Write JSON + Markdown under `results/<topic>/`.
2. Link from [`results/README.md`](results/README.md).
3. Update the relevant [model card](model_cards/) if it is a released checkpoint.
4. For major findings, add a section to [`docs/IJEPA_Looped_Predictor_Report.md`](docs/IJEPA_Looped_Predictor_Report.md).

## Questions

Open an issue with: config used, command run, expected vs actual behavior, and relevant
log excerpts from `runs/`.
