#!/usr/bin/env bash
# Hanya random search berpusat di baseline + re-run pemenang top-1 di full seeds × profiles.
# Tanpa baseline. (Nama file lama dipertahankan untuk kompatibilitas.)
#
# Default: N=16 trial, seed sampling=99, search di seed=0 × profile=minimal;
# pemenang top-1 di-rerun di --seeds × --profiles (default 3 × 2 = 6 run).
#
# Dari akar repositori: ./scripts/run_hyperband_only.sh
# Argumen tambahan diteruskan ke run_experiment.py.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec python3 scripts/run_experiment.py \
  --search-only \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --random-search-n 16 \
  --random-search-seed 99 \
  --random-search-sigma 1.0 \
  --search-train-seed 0 \
  --search-profile minimal \
  --cv-seed 12345 \
  "$@"
