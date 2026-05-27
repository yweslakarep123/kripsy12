#!/usr/bin/env bash
# Hanya dua fase random search (epoch 5000 lalu 3000), tanpa baseline.
#
# Default: seed=101, profil=standard, N=10 trial/fase, sigma=1.0.
# Pilih seed dari baseline: --search-train-seed -1
#
# Dari akar repositori: ./scripts/run_search_only.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/run_experiment.py \
  --search-only \
  --seeds 0 42 101 \
  --search-train-seed 101 \
  --search-profile standard \
  --random-search-n 10 \
  --random-search-seed 99 \
  --random-search-sigma 1.0 \
  --search-max-batch-size 512 \
  --cv-seed 12345 \
  "$@"
