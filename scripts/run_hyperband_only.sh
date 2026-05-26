#!/usr/bin/env bash
# Hanya dua fase random search (epoch~5000 lalu epoch~3000).
#
# Seed training: otomatis dari baseline terbaik di results.csv,
# atau override dengan --search-train-seed.
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
  --random-search-n 10 \
  --random-search-seed 99 \
  --random-search-sigma 1.0 \
  --search-max-batch-size 512 \
  --cv-seed 12345 \
  "$@"
