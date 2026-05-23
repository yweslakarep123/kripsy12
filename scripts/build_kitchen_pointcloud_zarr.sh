#!/usr/bin/env bash
# Bangun zarr Kitchen DENGAN point_cloud (512 pts) dari Minari D4RL.
# Wajib dijalankan sekali per mesin (Vast.ai / laptop) sebelum baseline atau random search.
#
# Minari mengunduh dataset D4RL/kitchen/complete-v2 otomatis (~beberapa menit pertama kali).
# Keluaran: FlowPolicy/FlowPolicy/data/kitchen_complete_from_minari.zarr
#
# Dari akar repo:
#   conda activate flowpolicy-kitchen
#   ./scripts/build_kitchen_pointcloud_zarr.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate flowpolicy-kitchen 2>/dev/null || true
fi

OUT="${ZARR_OUT:-FlowPolicy/data/kitchen_complete_from_minari.zarr}"
DEVICE="${ZARR_DEVICE:-cuda:0}"
SAMPLING="${ZARR_SAMPLING:-fps}"

echo ">>> Ekspor zarr point cloud (512 pts) -> ${OUT}"
echo "    Minari: D4RL/kitchen/complete-v2 | device=${DEVICE} | sampling=${SAMPLING}"
echo ""

python3 scripts/export_minari_kitchen_to_flowpolicy_zarr.py \
  --out "${OUT}" \
  --minari-id D4RL/kitchen/complete-v2 \
  --device "${DEVICE}" \
  --sampling "${SAMPLING}" \
  --num-points 512

echo ""
echo ">>> Verifikasi isi zarr..."
python3 - <<'PY'
import os
import sys
import zarr

repo = os.path.abspath(".")
# Path relatif export script (cwd=FlowPolicy): FlowPolicy/data/... 
candidates = [
    os.path.join(repo, "FlowPolicy", "FlowPolicy", "data", "kitchen_complete_from_minari.zarr"),
    os.path.join(repo, "FlowPolicy", "data", "kitchen_complete_from_minari.zarr"),
]
zpath = next((p for p in candidates if os.path.isdir(p)), None)
if zpath is None:
    print("ERROR: zarr tidak ditemukan setelah ekspor.", file=sys.stderr)
    sys.exit(1)
root = zarr.open(zpath, "r")
keys = list(root["data"].keys())
print(f"  path: {zpath}")
print(f"  keys: {keys}")
if "point_cloud" not in keys:
    print("ERROR: point_cloud masih tidak ada — jangan pakai --no-point-cloud.", file=sys.stderr)
    sys.exit(1)
pc = root["data"]["point_cloud"]
print(f"  point_cloud shape: {pc.shape}")
print("OK: zarr siap untuk training point cloud.")
PY
