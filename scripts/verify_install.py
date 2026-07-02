#!/usr/bin/env python3
"""Verify a fresh clone: imports, configs, tests, and optional checkpoints.

Usage::

    python scripts/verify_install.py
    python scripts/verify_install.py --require-checkpoints

Exit code 0 = ready to work. Non-zero = fix the reported issue before training.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_CONFIGS = (
    "configs/image_jepa_cifar10_v3.yaml",
    "configs/image_jepa_cifar10_v3_looped.yaml",
    "configs/ablations/base_v3_looped.yaml",
)

OPTIONAL_CHECKPOINTS = (
    "checkpoints/baseline_v3/latest.pt",
    "checkpoints/baseline_v3_looped/latest.pt",
)


def _check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f": {detail}"
    print(line)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify install after fresh clone")
    parser.add_argument(
        "--require-checkpoints",
        action="store_true",
        help="Fail if pretrained checkpoints are missing",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    print(f"Repository root: {ROOT}")
    ok = True

    sys.path.insert(0, str(ROOT / "src"))
    try:
        import jepa  # noqa: F401
        import torch  # noqa: F401
        ok &= _check("import jepa", True, f"torch {torch.__version__}")
    except ImportError as exc:
        ok &= _check("import jepa", False, str(exc))

    for rel in REQUIRED_CONFIGS:
        path = ROOT / rel
        ok &= _check(f"config {rel}", path.exists())

    for rel in OPTIONAL_CHECKPOINTS:
        path = ROOT / rel
        exists = path.exists()
        if args.require_checkpoints:
            ok &= _check(f"checkpoint {rel}", exists)
        else:
            _check(f"checkpoint {rel}", True, "present" if exists else "missing (train or copy weights)")

    if not args.skip_tests:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_shapes.py", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        ok &= _check("pytest tests/test_shapes.py", r.returncode == 0, r.stdout.strip() or r.stderr.strip())

    if ok:
        print("\nReady. Next steps:")
        print("  python scripts/linear_probe.py --config configs/image_jepa_cifar10_v3.yaml \\")
        print("    --checkpoint checkpoints/baseline_v3/latest.pt   # if checkpoint present")
        print("  python visualizations/generate_all_figures.py --fast")
        print("  uv sync --extra demo && python app.py")
    else:
        print("\nFix failures above. See REPRODUCTION.md and CONTRIBUTING.md.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
