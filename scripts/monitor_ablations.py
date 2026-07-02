#!/usr/bin/env python3
"""Monitor ablation training; finalize reports when all variants complete."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LOG = PROJECT / "runs/ablations_train.log"
SUMMARY = PROJECT / "results/ablations/summary.json"
DESKTOP_REVIEW = Path.home() / "Desktop/jepa_ablation_performance_review.md"
DESKTOP_MAIN = Path.home() / "Desktop/jepa_looped_v3_performance_review.md"


def _proc_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "run_ablations.py"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def _latest_epoch() -> int | None:
    import re

    if not LOG.exists():
        return None
    epochs = re.findall(r"epoch (\d+)/300", LOG.read_text(errors="ignore"))
    return int(epochs[-1]) if epochs else None


def _current_variant() -> str | None:
    import re

    if not LOG.exists():
        return None
    hits = re.findall(r"=== (\S+ / \S+) ===", LOG.read_text(errors="ignore"))
    return hits[-1] if hits else None


def _all_variants_complete() -> bool:
    if not SUMMARY.exists():
        return False
    data = json.loads(SUMMARY.read_text())
    for suite in data.get("suites", []):
        for variant in suite.get("variants", []):
            if variant.get("status") == "missing_checkpoint":
                return False
            if "top1_accuracy" not in variant:
                return False
    return len(data.get("suites", [])) >= 3


def _finalize() -> None:
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts/run_ablations.py"), "--suite", "all", "--eval-only"],
        cwd=PROJECT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts/generate_looped_performance_review.py"),
            "--skip-compare",
            "--out",
            str(DESKTOP_MAIN),
        ],
        cwd=PROJECT,
        check=True,
    )
    summary_md = PROJECT / "results/ablations/summary.md"
    comparison = PROJECT / "runs/looped_v3_comparison.json"
    lines = [
        "# I-JEPA Looped Predictor: Ablation Performance Review",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "Full ablation training (v3 recipe, 300 epochs) is complete.",
        "",
        "## v3 baseline reference",
        "",
        "| Model | Tuned top-1 | feat_std |",
        "| --- | --- | --- |",
        "| v3 baseline | **77.23%** | 0.1607 |",
        "",
    ]
    if comparison.exists():
        comp = json.loads(comparison.read_text())
        lines.extend(
            [
                "## Head-to-head (2-loop default vs v3)",
                "",
                f"- v3 baseline: **{comp['baseline']['top1_accuracy']:.2f}%**",
                f"- v3 + looped (2 loops): **{comp['looped']['top1_accuracy']:.2f}%** "
                f"({comp['delta_top1']:+.2f} pp)",
                "",
            ]
        )
    if summary_md.exists():
        lines.append(summary_md.read_text().strip())
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            f"- Detailed results: `{PROJECT / 'results/ablations/'}`",
            f"- Training log: `{LOG}`",
            "",
        ]
    )
    DESKTOP_REVIEW.write_text("\n".join(lines))
    print(f"Wrote {DESKTOP_REVIEW}")
    print(f"Updated {DESKTOP_MAIN}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-if-done", action="store_true")
    args = parser.parse_args()

    running = _proc_running()
    epoch = _latest_epoch()
    variant = _current_variant()
    complete = _all_variants_complete() and not running

    print(f"running={running} complete={complete} epoch={epoch} variant={variant}")

    if complete:
        if args.finalize_if_done:
            _finalize()
        print("STATUS: DONE")
        sys.exit(0)

    if not running and not complete:
        print("STATUS: STOPPED_INCOMPLETE")
        sys.exit(2)

    print("STATUS: IN_PROGRESS")
    sys.exit(1)


if __name__ == "__main__":
    main()
