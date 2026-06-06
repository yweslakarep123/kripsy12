#!/usr/bin/env bash
# Hyperband + rerun pemenang, target ≤ ~48 jam pada GPU ~30 TFLOPS efektif.
# (Tanpa baseline; 3 seed × 2 profil per trial; resume antar-rung aktif.)
#
# Kalibrasi: 1 run penuh 3000 epoch × 1 seed × 1 profil ≈ 2,7 jam @ 30 TFLOPS.
# Jika lebih lambat, turunkan --hyperband-max-epochs atau naikkan --hyperband-s-min.
#
# Dari akar repositori:
#   ./scripts/run_twoday_30tflops.sh
#   ./scripts/run_twoday_30tflops.sh --output-dir outputs/my_twoday_30

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec python3 scripts/run_experiment.py \
  --hyperband-only \
  --output-dir "${OUTPUT_DIR:-outputs/twoday_30tflops}" \
  --zarr-path "${ZARR_PATH:-FlowPolicy/data/kitchen_complete_from_minari.zarr}" \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --hyperband-max-epochs 1200 \
  --hyperband-eta 3 \
  --hyperband-s-max 2 \
  --hyperband-s-min 2 \
  --hyperband-seed 99 \
  --hyperband-sampling baseline_anchored \
  --hyperband-iterations 1 \
  --max-batch-size 128 \
  --dataloader-num-workers 2 \
  "$@"
