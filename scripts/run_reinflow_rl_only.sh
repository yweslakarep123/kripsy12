#!/usr/bin/env bash
# Hanya fine-tuning RL online (ReinFlow-style) tanpa baseline.
# Butuh checkpoint BC di runs/baseline_seed{seed}_{profile}/checkpoints/latest.ckpt
# di bawah --output-dir yang sama.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec python3 scripts/run_experiment.py \
  --reinflow-rl-only \
  --hyperparam-search reinflow \
  "$@"
