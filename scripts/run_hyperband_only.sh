#!/usr/bin/env bash
# Alias legacy: dulu Hyperband, sekarang random search saja.
# Lihat scripts/run_search_only.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec "$ROOT/scripts/run_search_only.sh" "$@"
