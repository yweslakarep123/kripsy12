#!/usr/bin/env python3
"""Random search berpusat di baseline untuk hiperparameter FlowPolicy Kitchen.

Menggantikan Hyperband: setiap trial dilatih penuh sampai ``training.num_epochs``
(yang disample dari ``LOCAL_SEARCH_SPACE`` dengan bobot di sekitar baseline).
Pemenang dipilih berdasarkan ``val_loss_final`` terkecil (sama seperti sinyal
Hyperband). State persisten: ``<output-dir>/random_search_state.json``.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_constants import (  # noqa: E402
    CSV_HPARAM_KEYS,
    DEFAULT_BASELINE_HPARAMS,
    SEARCH_CFG_IDX_BASE,
    sample_configs_around_baseline,
)
from hyperband_search import (  # noqa: E402
    _evaluate_config_at_rung,
    _run_dir_for_cfg,
)


def _state_path(out_root: pathlib.Path) -> pathlib.Path:
    return out_root / "random_search_state.json"


def _load_state(out_root: pathlib.Path) -> Optional[Dict[str, Any]]:
    p = _state_path(out_root)
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _save_state(out_root: pathlib.Path, state: Dict[str, Any]) -> None:
    p = _state_path(out_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, p)


def _pick_best_from_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    evals = state.get("evaluations", [])
    if not evals:
        return None

    def keyfn(e: Dict[str, Any]) -> float:
        v = e.get("val_loss")
        if v is None:
            return float("inf")
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return float("inf")
        if math.isnan(vf) or math.isinf(vf):
            return float("inf")
        return vf

    ordered = sorted(evals, key=keyfn)
    best = ordered[0]
    if keyfn(best) == float("inf"):
        return None

    cfg_idx = int(best["cfg_idx"])
    for cstate in state.get("configs", []):
        if int(cstate["cfg_idx"]) == cfg_idx:
            return {
                "cfg_idx": cfg_idx,
                "hparams": dict(cstate["hparams"]),
                "val_loss": float(best["val_loss"]),
            }
    return None


def run_random_search(
    *,
    out_root: pathlib.Path,
    runs_root: pathlib.Path,
    n_trials: int,
    sampling_seed: int,
    search_train_seed: int,
    search_profile: str,
    train_eps: List[int],
    val_eps: List[int],
    zarr_rel: str,
    checkpoint_every: int,
    dataloader_num_workers: int,
    py: str,
    train_py: pathlib.Path,
    cwd_train: str,
    apply_vram_limits_fn: Callable[[Dict[str, Any], int], Dict[str, Any]],
    max_batch_size: int,
    center_hparams: Optional[Dict[str, Any]] = None,
    sigma: float = 1.0,
    p_exact_baseline: float = 0.15,
    enable_early_stop: bool = True,
    early_stop_rollout_every: int = 200,
) -> Optional[Dict[str, Any]]:
    """Jalankan random search dan kembalikan konfigurasi pemenang."""
    out_root = pathlib.Path(out_root).resolve()
    runs_root = pathlib.Path(runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    n_trials = int(n_trials)
    center = center_hparams or dict(DEFAULT_BASELINE_HPARAMS)

    state = _load_state(out_root)
    reuse = False
    if state is not None:
        reuse = (
            int(state.get("n_trials", -1)) == n_trials
            and int(state.get("sampling_seed", -1)) == int(sampling_seed)
            and int(state.get("search_train_seed", -1)) == int(search_train_seed)
            and str(state.get("search_profile", "")) == str(search_profile)
        )
        if not reuse:
            print(
                "[random_search] parameter berubah vs random_search_state.json — "
                "membuat state baru."
            )

    if not reuse:
        rng = np.random.RandomState(int(sampling_seed))
        cfgs = sample_configs_around_baseline(
            rng,
            n_trials,
            center=center,
            base_cfg_idx=SEARCH_CFG_IDX_BASE,
            sigma=sigma,
            p_exact_baseline=p_exact_baseline,
        )
        cfgs = [apply_vram_limits_fn(c, max_batch_size) for c in cfgs]
        configs_state = []
        for cfg in cfgs:
            configs_state.append(
                {
                    "cfg_idx": int(cfg["cfg_idx"]),
                    "hparams": {k: cfg[k] for k in CSV_HPARAM_KEYS if k in cfg},
                    "done": False,
                    "epoch_trained": 0,
                }
            )
        state = {
            "version": 1,
            "algorithm": "random_search_around_baseline",
            "n_trials": n_trials,
            "sampling_seed": int(sampling_seed),
            "search_train_seed": int(search_train_seed),
            "search_profile": str(search_profile),
            "center_hparams": {k: center[k] for k in CSV_HPARAM_KEYS if k in center},
            "sigma": float(sigma),
            "p_exact_baseline": float(p_exact_baseline),
            "configs": configs_state,
            "evaluations": [],
            "best": None,
        }
        _save_state(out_root, state)

    for cstate in state["configs"]:
        if cstate.get("done"):
            continue
        cfg_idx = int(cstate["cfg_idx"])
        cfg = dict(cstate["hparams"])
        cfg["cfg_idx"] = cfg_idx
        run_dir = _run_dir_for_cfg(
            runs_root, cfg_idx, search_train_seed, search_profile
        )
        target_epochs = int(cfg["training.num_epochs"])
        already_trained = int(cstate.get("epoch_trained", 0))

        if cstate.get("done") and (run_dir / "training_final.json").is_file():
            continue

        if already_trained >= target_epochs and (run_dir / "training_final.json").is_file():
            cstate["done"] = True
            continue

        val_loss, rc, epoch_trained = _evaluate_config_at_rung(
            cfg=cfg,
            target_epoch=target_epochs,
            already_trained=already_trained,
            run_dir=run_dir,
            py=py,
            train_py=train_py,
            cwd_train=cwd_train,
            seed=search_train_seed,
            profile=search_profile,
            train_eps=train_eps,
            val_eps=val_eps,
            zarr_rel=zarr_rel,
            checkpoint_every=checkpoint_every,
            dataloader_num_workers=dataloader_num_workers,
            enable_early_stop=enable_early_stop,
            early_stop_rollout_every=early_stop_rollout_every,
        )
        if rc == 0:
            cstate["epoch_trained"] = int(epoch_trained)
        cstate["done"] = True
        state["evaluations"].append(
            {
                "cfg_idx": cfg_idx,
                "val_loss": val_loss,
                "returncode": rc,
            }
        )
        _save_state(out_root, state)
        print(
            f"[random_search] cfg_idx={cfg_idx} selesai "
            f"val_loss={val_loss} rc={rc}"
        )

    best = _pick_best_from_state(state)
    if best is not None:
        state["best"] = best
        _save_state(out_root, state)
        print(
            f"[random_search] pemenang cfg_idx={best['cfg_idx']} "
            f"val_loss={best['val_loss']:.6f}"
        )
    return best
