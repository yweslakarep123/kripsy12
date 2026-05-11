#!/usr/bin/env bash
# Hanya random search (tanpa baseline).
# Urutan seed per kombinasi (cfg × profil): 0 → 42 → 1010 → 0, lalu berulang (4 posisi per siklus).
# Sampling hiperparameter RS: acak per eksekusi skrip kecuali SAMPLING_SEED diset (atau configs.json sudah ada — akan dipakai ulang).
#
# Dari akar repositori: ./scripts/run_experiment_random_search.sh
# Argumen tambahan diteruskan ke run_experiment.py, mis. --n-configs 5 --output-dir outputs/rs1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Seed sampling random search (reproduksi grid): override dengan SAMPLING_SEED=... di lingkungan.
if [[ -z "${SAMPLING_SEED+x}" ]]; then
  SAMPLING_SEED=$((RANDOM * 32768 + RANDOM))
fi

exec python3 scripts/run_experiment.py \
  --random-search-only \
  --seeds 0 42 1010 0 \
  --profiles standard minimal \
  --n-configs 10 \
  --sampling-seed "${SAMPLING_SEED}" \
  --cv-seed 12345 \
  "$@"
