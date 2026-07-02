#!/usr/bin/env python3
"""Deprecated entry point; the demo now lives at the repo root as ``app.py``.

Kept for backwards compatibility; forwards to ``app.main()``.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    print("[gradio_demo] Deprecated: launching the new demo (app.py). "
          "Use `python app.py` directly.")
    from app import main

    main()
