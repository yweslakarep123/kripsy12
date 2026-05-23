#!/usr/bin/env bash
# Hanya random search berpusat di baseline + re-run pemenang top-1.
#
# Fase SEARCH (N trial): 1 seed × profil minimal SAJA (--search-profile minimal).
# Fase RERUN pemenang: --seeds × --profiles (default: 3 seed × 1 profil minimal).
#
# WAJIB: zarr dengan point_cloud — jalankan dulu:
#   ./scripts/build_kitchen_pointcloud_zarr.sh
#
# Dari akar repositori: ./scripts/run_hyperband_only.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec python3 scripts/run_experiment.py \
  --search-only \
  --seeds 0 42 101 \
  --profiles minimal \
  --random-search-n 16 \
  --random-search-seed 99 \
  --random-search-sigma 1.0 \
  --search-train-seed 0 \
  --search-profile minimal \
  --search-max-batch-size 512 \
  --cv-seed 12345 \
  "$@"
