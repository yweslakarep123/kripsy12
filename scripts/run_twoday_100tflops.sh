#!/usr/bin/env bash
# Pipeline ringkas target ≤ ~48 jam pada GPU ~100 TFLOPS efektif:
# baseline (6 run) + Hyperband (bracket s=2..0) + rerun pemenang (6 run).
#
# Kalibrasi: 1 run penuh 3000 epoch × 1 seed × 1 profil ≈ 0,8 jam @ 100 TFLOPS.
# Jika mendekati 48 jam, cap bracket: --hyperband-s-max 2 --hyperband-s-min 2.
#
# Dari akar repositori:
#   ./scripts/run_twoday_100tflops.sh
#   ./scripts/run_twoday_100tflops.sh --output-dir outputs/my_twoday_100

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec python3 scripts/run_experiment.py \
  --output-dir "${OUTPUT_DIR:-outputs/twoday_100tflops}" \
  --zarr-path "${ZARR_PATH:-FlowPolicy/data/kitchen_complete_from_minari.zarr}" \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --hyperband-max-epochs 3000 \
  --hyperband-eta 3 \
  --hyperband-s-max 2 \
  --hyperband-s-min 0 \
  --hyperband-seed 99 \
  --hyperband-sampling baseline_anchored \
  --hyperband-iterations 1 \
  --max-batch-size 128 \
  --dataloader-num-workers 4 \
  "$@"
