#!/usr/bin/env bash
# Hanya Hyperband (Li et al., 2018) + re-run pemenang top-1 di full seeds × profiles.
# Tanpa baseline.
#
# Default: R=3000 (= baseline num_epochs), eta=3 (sesuai paper).
# Hyperband mengevaluasi tiap trial di --seeds × --profiles (default 3×2);
# pemenang top-1 di-rerun train+infer di kombinasi yang sama (6 run).
#
# Anggaran waktu (perkiraan; lihat README untuk detail):
#   --hyperband-s-min 0 (default): SEMUA bracket s=s_max..0  → paling robust,
#       paling mahal (bisa > 2 hari untuk R=3000 tergantung kecepatan training).
#   --hyperband-s-min 2: single-bracket SHA s=2 saja             → tercepat,
#       sesuai paper "bracket s=k as a standalone SuccessiveHalving".
#
# Dari akar repositori: ./scripts/run_hyperband_only.sh
# Argumen tambahan diteruskan ke run_experiment.py.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec python3 scripts/run_experiment.py \
  --hyperband-only \
  --seeds 0 42 101 \
  --profiles standard minimal \
  --hyperband-max-epochs 3000 \
  --hyperband-eta 3 \
  --hyperband-s-min 0 \
  --hyperband-seed 99 \
  --cv-seed 12345 \
  "$@"
