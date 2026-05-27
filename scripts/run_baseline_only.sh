#!/usr/bin/env bash
# Hanya 3 pelatihan baseline: 3 seed × profil minimal (tanpa augmentasi noise).
# Dari akar repositori: ./scripts/run_baseline_only.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/run_experiment.py \
  --baseline-only \
  --seeds 0 42 101 \
  --cv-seed 12345 \
  "$@"
