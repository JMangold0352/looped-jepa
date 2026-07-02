#!/bin/bash
# Poll ablation training every 3 hours; finalize Desktop reviews when complete.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

while true; do
  sleep 10800
  echo "AGENT_LOOP_TICK_ablations $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python scripts/monitor_ablations.py --finalize-if-done || true
done
