#!/usr/bin/env python3
"""Train and evaluate looped-predictor ablation suites (v3 recipe, 300 epochs).

Suites:
  loop_count     : 1 / 2 / 4 recurrence steps
  entropy        : exit-gate entropy reg on vs off
  sandwich_norm  : LayerNorm vs sandwich RMSNorm in the predictor

Results land in ``results/ablations/`` as JSON + Markdown tables.

Examples::

    # Evaluate loops_2 from the existing v3-looped checkpoint; train the rest:
    python scripts/run_ablations.py --suite all --train

    # Re-run reporting only (checkpoints must exist):
    python scripts/run_ablations.py --suite all --eval-only

    # Smoke test (1 epoch) during development:
    python scripts/run_ablations.py --suite loop_count --train --epochs 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from jepa.ablations.registry import ABLATION_SUITES, suite_names
from jepa.ablations.report import save_suite_results, save_summary
from jepa.ablations.runner import run_suite
from jepa.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run looped-predictor ablation suites")
    parser.add_argument(
        "--suite",
        choices=[*suite_names(), "all"],
        default="all",
        help="Which ablation suite to run",
    )
    parser.add_argument(
        "--config",
        default="configs/ablations/base_v3_looped.yaml",
        help="Base config (v3 training recipe + looped predictor)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/ablations",
        help="Directory for JSON + Markdown results",
    )
    parser.add_argument("--train", action="store_true", help="Train variants (default: eval only)")
    parser.add_argument("--eval-only", action="store_true", help="Alias for omitting --train")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count (default: 300)")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-train even when a complete checkpoint already exists",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume partial checkpoints (train from scratch)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_run = args.train and not args.eval_only
    device = get_device("auto")
    print(f"Device: {device}")
    print(f"Mode: {'train+eval' if train_run else 'eval-only'}")

    suite_keys = suite_names() if args.suite == "all" else [args.suite]
    out_dir = Path(args.out_dir)
    all_payloads = []

    for key in suite_keys:
        suite = ABLATION_SUITES[key]
        payload = run_suite(
            suite,
            args.config,
            device,
            train_run=train_run,
            skip_existing=not args.no_skip_existing,
            epochs=args.epochs,
            resume=not args.no_resume,
        )
        save_suite_results(payload, out_dir)
        all_payloads.append(payload)

    save_summary(all_payloads, out_dir)
    print(f"\nDone. Results in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
