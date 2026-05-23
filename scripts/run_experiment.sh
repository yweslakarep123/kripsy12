#!/usr/bin/env bash
# Eksperimen penuh: (1) baseline lalu (2) random search berpusat di baseline +
# (3) rerun pemenang top-1 di full seeds × profiles.
#
# Default random search: N=16 trial, sigma=1.0 (lebih sering sample di sekitar baseline).
#
# Dari akar repositori: ./scripts/run_experiment.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/run_experiment.py \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --random-search-n 16 \
  --random-search-seed 99 \
  --random-search-sigma 1.0 \
  --search-train-seed 0 \
  --search-profile minimal \
  --cv-seed 12345 \
  "$@"
