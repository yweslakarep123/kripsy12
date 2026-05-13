#!/usr/bin/env bash
# Eksperimen penuh: baseline lalu pencarian hiperparameter (default: optimasi Bayesian).
# Dari akar repositori: ./scripts/run_experiment.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/run_experiment.py \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --n-configs 10 \
  --sampling-seed 99 \
  --cv-seed 12345 \
  "$@"
