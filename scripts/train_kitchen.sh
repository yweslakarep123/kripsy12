#!/usr/bin/env bash
# Contoh:
#   bash scripts/train_kitchen.sh franka_kitchen_microwave 42 0
#   bash scripts/train_kitchen.sh franka_kitchen_sequential4 42 0
set -euo pipefail
TASK="${1:-franka_kitchen_microwave}"
SEED="${2:-42}"
GPU="${3:-0}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}/FlowPolicy"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${GPU}"

python train.py --config-name=flowpolicy.yaml \
  task="${TASK}" \
  training.seed="${SEED}" \
  training.device="cuda:0" \
  exp_name="${TASK}-flowpolicy-seed${SEED}"
