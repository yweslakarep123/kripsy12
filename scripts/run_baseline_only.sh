#!/usr/bin/env bash
# Hanya 6 pelatihan baseline: 3 seed × 2 profil (standard, minimal).
# Kitchen lowdim 7-task — eval multi-seed 0,42,101 via infer_kitchen_lowdim.py.
# Dari akar repositori: ./scripts/run_baseline_only.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATASET_DIR="${DATASET_DIR:-FlowPolicy/data/kitchen/kitchen_demos_multitask}"
exec python3 scripts/run_experiment.py \
  --baseline-only \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --dataset-dir "$DATASET_DIR" \
  --cv-seed 12345 \
  --n-infer-episodes 50 \
  "$@"
