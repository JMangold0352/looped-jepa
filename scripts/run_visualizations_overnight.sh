#!/usr/bin/env bash
# Generate all publication figures overnight (safe sample caps, logs to runs/).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

python -m pip install -q scikit-learn 2>/dev/null || true

LOG="runs/visualizations_generate.log"
mkdir -p runs visualizations/figures

echo "=== visualization run started $(date) ===" | tee -a "$LOG"
python visualizations/generate_all_figures.py 2>&1 | tee -a "$LOG"
echo "=== visualization run finished $(date) ===" | tee -a "$LOG"
