#!/usr/bin/env bash
# Eksperimen penuh: (1) baseline 3 seed → (2) pilih seed terbaik →
# (3) random search epoch~5000 → (4) random search epoch~3000.
#
# Default random search: N=10 trial/fase, sigma=1.0.
#
# Dari akar repositori: ./scripts/run_experiment.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/run_experiment.py \
  --seeds 0 42 101 \
  --random-search-n 10 \
  --random-search-seed 99 \
  --random-search-sigma 1.0 \
  --search-max-batch-size 512 \
  --cv-seed 12345 \
  "$@"
