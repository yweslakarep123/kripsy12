#!/usr/bin/env bash
# Uji coba random search di laptop (VRAM ~8 GB) — BUKAN run produksi.
#
# Tujuan: memastikan orkestrator + train.py + random_search_state.json jalan
# end-to-end dengan biaya waktu kecil (N=2 trial, profil minimal, 1 seed rerun).
#
# Prasyarat:
#   conda activate flowpolicy-kitchen
#   PyTorch harus melihat GPU: python -c "import torch; assert torch.cuda.is_available()"
#   Dataset zarr ada (default path di bawah).
#
# Dari akar repo:
#   ./scripts/run_hyperband_laptop_smoke.sh
#
# Jika CUDA error "unknown error": reboot laptop, tutup app GPU berat, lalu coba lagi.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda tidak ditemukan di PATH."
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowpolicy-kitchen

echo ">>> Cek CUDA..."
python -c "import torch; assert torch.cuda.is_available(), (
    'CUDA tidak tersedia. Perbaiki driver/GPU dulu (reboot, nvidia-smi, lalu ulang).'
); print('OK GPU:', torch.cuda.get_device_name(0))"

ZARR="${ZARR_PATH:-FlowPolicy/data/kitchen_complete_from_minari.zarr}"
OUT="${OUTPUT_DIR:-outputs/laptop_hyperband_smoke}"

exec python scripts/run_experiment.py \
  --search-only \
  --output-dir "$OUT" \
  --zarr-path "$ZARR" \
  --seeds 0 \
  --profiles minimal \
  --random-search-n 2 \
  --random-search-seed 7 \
  --search-train-seed 0 \
  --search-profile minimal \
  --n-infer-episodes 2 \
  --n-train-val-episodes 0 \
  --max-batch-size 16 \
  --dataloader-num-workers 0 \
  --checkpoint-every 1 \
  --skip-inference-videos \
  "$@"
