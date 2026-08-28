#!/usr/bin/env bash
# Download released pretrained checkpoints into checkpoints/ (idempotent).
#
# Usage:
#   ./scripts/download_weights.sh --list
#   ./scripts/download_weights.sh
#   ./scripts/download_weights.sh baseline_v3
#
# URLs: released_weights/urls.yaml or LOOPED_JEPA_WEIGHT_URL_* env vars.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv/bin" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python scripts/download_weights.py "$@"
