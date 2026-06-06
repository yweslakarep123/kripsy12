#!/usr/bin/env bash
# Eksperimen penuh: (1) baseline 3 seed → (2) random search epoch=5000 →
# (3) random search epoch=3000.
#
# Random search default: seed=101, profil=standard.
# Epoch random search: hanya 5000 dan 3000 (satu nilai tetap per fase).
#
# Dari akar repositori: ./scripts/run_experiment.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/run_experiment.py \
  --seeds 0 42 101 \
  --search-train-seed 101 \
  --search-profile standard \
  --random-search-n 10 \
  --random-search-seed 99 \
  --random-search-sigma 1.0 \
  --search-max-batch-size 512 \
  --cv-seed 12345 \
  "$@"
