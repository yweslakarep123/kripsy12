#!/usr/bin/env bash
# Verifikasi logika Hyperband TANPA GPU (mock training).
# Berguna jika CUDA di laptop sedang bermasalah tetapi Anda ingin memastikan
# kode orkestrator + rumus paper (Tabel 1) benar.
#
#   conda activate flowpolicy-kitchen
#   ./scripts/verify_hyperband_no_gpu.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/scripts"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate flowpolicy-kitchen 2>/dev/null || true
fi

python3 << 'PY'
import json, math, pathlib, tempfile
from unittest.mock import patch
import hyperband_search as hb
from hyperband_search import compute_brackets, run_hyperband, _top_k_by_val_loss
from experiment_constants import (
    sample_configs_hyperband,
    append_kitchen_policy_hparam_overrides,
    baseline_search_center,
    HYPERBAND_SAMPLING_BASELINE_ANCHORED,
)
from hyperband_search import _build_train_overrides_hb
import numpy as np

rng = np.random.RandomState(42)
center = baseline_search_center()
cfgs = sample_configs_hyperband(rng, 5, base_cfg_idx=1000, sampling=HYPERBAND_SAMPLING_BASELINE_ANCHORED)
assert cfgs[0]["cfg_idx"] == 1000
for k in center:
    assert cfgs[0][k] == center[k], (k, cfgs[0][k], center[k])
for c in cfgs[1:]:
    diffs = [k for k in center if c[k] != center[k]]
    assert 1 <= len(diffs) <= 4, diffs
print("OK: baseline_anchored sampling (baseline + tweak lokal)")

# State encoder + override Hydra
cfg = {"cfg_idx": 1000, "training.num_epochs": 0, "optimizer.lr": 1e-4,
       "dataloader.batch_size": 128, "policy.Conditional_ConsistencyFM.num_segments": 2,
       "policy.Conditional_ConsistencyFM.eps": 1e-2, "policy.Conditional_ConsistencyFM.delta": 1e-2,
       "n_action_steps": 4, "n_obs_steps": 2, "policy.diffusion_step_embed_dim": 128,
       "_state_mlp_hidden": 256}
odl = _build_train_overrides_hb(cfg, seed=0, profile="standard", train_eps=[0], val_eps=[1],
    run_dir=pathlib.Path("/tmp/hb"), zarr_rel="x.zarr", resume_training=False,
    delta_num_epochs=3, checkpoint_every=1, dataloader_num_workers=0)
assert "policy.obs_encoder_type=state" in odl
assert "task.env_runner.obs_mode=state" in odl
assert "policy.encoder_output_dim=256" in odl
assert not any("point_cloud" in x for x in odl)
print("OK: override Hydra memakai state encoder")

# Tabel 1 paper R=81 eta=3
brs = compute_brackets(81, 3)
assert brs[0].s == 4 and brs[0].n == 81 and brs[0].rungs[-1].r_i == 81
print("OK: compute_brackets cocok Tabel 1 paper (R=81, eta=3)")

def fake_apply(cfg, mb):
    return dict(cfg)

def fake_eval(*, cfg, target_epoch, already_trained, run_dir, **kw):
    v = float(cfg["optimizer.lr"]) + 0.01 * int(cfg["cfg_idx"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "checkpoints" / "latest.ckpt").write_text("x")
    (run_dir / "training_final.json").write_text(json.dumps({"val_loss_final": v}))
    return v, 0, int(target_epoch)

with tempfile.TemporaryDirectory() as d:
    out, runs = pathlib.Path(d), pathlib.Path(d) / "runs"
    with patch.object(hb, "_evaluate_config_at_rung", side_effect=fake_eval):
        best = run_hyperband(
            out_root=out, runs_root=runs, R=12, eta=2, s_min=1, s_max=1,
            hyperband_iterations=2,
            sampling=HYPERBAND_SAMPLING_BASELINE_ANCHORED,
            sampling_seed=7,
            search_seeds=[0, 42],
            search_profiles=["standard", "minimal"],
            train_eps=[0,1], val_eps=[2], zarr_rel="x.zarr",
            checkpoint_every=1, dataloader_num_workers=0,
            py="python3", train_py=pathlib.Path("/tmp/t.py"), cwd_train="/tmp",
            apply_vram_limits_fn=fake_apply, max_batch_size=128,
        )
    st = json.loads((out / "hyperband_state.json").read_text())
    assert int(st.get("hyperband_iterations", 0)) == 2
    assert len(st.get("iterations", [])) == 2
    assert best and (out / "hyperband_state.json").is_file()
    print(f"OK: run_hyperband end-to-end, pemenang cfg_idx={best['cfg_idx']}")
print("\nSemua verifikasi logika Hyperband LULUS (tanpa GPU).")
PY
