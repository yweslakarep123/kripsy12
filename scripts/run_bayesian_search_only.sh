#!/usr/bin/env bash
# Hanya optimasi Bayesian (GP + EI), tanpa baseline.
# Jumlah trial ≈ --n-configs (default 10) × jumlah seed × jumlah profil.
#
# Dari akar repositori: ./scripts/run_bayesian_search_only.sh
# Argumen tambahan diteruskan ke run_experiment.py, mis. --n-configs 5 --output-dir outputs/bo1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec python3 scripts/run_experiment.py \
  --bayesian-search-only \
  --hyperparam-search bayesian \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --n-configs 10 \
  --sampling-seed 99 \
  --cv-seed 12345 \
  "$@"
