#!/usr/bin/env python3
"""Random search berpusat di baseline untuk hiperparameter FlowPolicy Kitchen.

Setiap trial dilatih + inferensi (override Hydra identik baseline).
Pemenang = ``test_success_rate_total`` terbesar.
State per fase: ``random_search_state_epoch5000.json`` / ``..._epoch3000.json``.
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
    EPOCH_SEARCH_CFG_IDX_BASE,
    EPOCH_SEARCH_STATE_FILES,
    sample_configs_around_baseline,
)
from search_training import (  # noqa: E402
    run_dir_for_search_cfg,
    run_search_trial,
)


def _state_path(out_root: pathlib.Path, epoch_mode: str) -> pathlib.Path:
    fname = EPOCH_SEARCH_STATE_FILES.get(
        epoch_mode, f"random_search_state_{epoch_mode}.json"
    )
    return out_root / fname


def _load_state(out_root: pathlib.Path, epoch_mode: str) -> Optional[Dict[str, Any]]:
    p = _state_path(out_root, epoch_mode)
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _save_state(out_root: pathlib.Path, epoch_mode: str, state: Dict[str, Any]) -> None:
    p = _state_path(out_root, epoch_mode)
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
        v = e.get("test_success_rate_total")
        if v is None:
            return float("-inf")
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return float("-inf")
        if math.isnan(vf) or math.isinf(vf):
            return float("-inf")
        return vf

    ordered = sorted(evals, key=keyfn, reverse=True)
    best = ordered[0]
    if keyfn(best) == float("-inf"):
        return None

    cfg_idx = int(best["cfg_idx"])
    for cstate in state.get("configs", []):
        if int(cstate["cfg_idx"]) == cfg_idx:
            return {
                "cfg_idx": cfg_idx,
                "hparams": dict(cstate["hparams"]),
                "test_success_rate_total": float(best["test_success_rate_total"]),
                "val_loss": best.get("val_loss"),
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
    infer_py: pathlib.Path,
    cwd_train: str,
    apply_vram_limits_fn: Callable[[Dict[str, Any], int], Dict[str, Any]],
    max_batch_size: int,
    center_hparams: Optional[Dict[str, Any]] = None,
    sigma: float = 1.0,
    p_exact_baseline: float = 0.15,
    enable_early_stop: bool = True,
    early_stop_rollout_every: int = 200,
    epoch_mode: str = "epoch_5000",
    results_csv: Optional[pathlib.Path] = None,
    hp_cols: Optional[List[str]] = None,
    fold_i: int = 0,
    n_infer_episodes: int = 50,
    n_train_val_episodes: int = 15,
    train_val_eval_seed_offset: int = 31,
    skip_inference_videos: bool = False,
) -> Optional[Dict[str, Any]]:
    """Jalankan random search satu fase dan kembalikan konfigurasi pemenang."""
    out_root = pathlib.Path(out_root).resolve()
    runs_root = pathlib.Path(runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    n_trials = int(n_trials)
    center = center_hparams or dict(DEFAULT_BASELINE_HPARAMS)
    cfg_idx_base = EPOCH_SEARCH_CFG_IDX_BASE.get(epoch_mode, 1000)

    state = _load_state(out_root, epoch_mode)
    reuse = False
    if state is not None:
        reuse = (
            int(state.get("n_trials", -1)) == n_trials
            and int(state.get("sampling_seed", -1)) == int(sampling_seed)
            and int(state.get("search_train_seed", -1)) == int(search_train_seed)
            and str(state.get("search_profile", "")) == str(search_profile)
            and str(state.get("epoch_mode", "")) == str(epoch_mode)
        )
        if not reuse:
            print(
                f"[random_search:{epoch_mode}] parameter berubah vs state — "
                "membuat state baru."
            )

    if not reuse:
        rng = np.random.RandomState(int(sampling_seed))
        cfgs = sample_configs_around_baseline(
            rng,
            n_trials,
            center=center,
            base_cfg_idx=cfg_idx_base,
            sigma=sigma,
            p_exact_baseline=p_exact_baseline,
            epoch_mode=epoch_mode,
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
            "version": 3,
            "algorithm": "random_search_around_baseline",
            "epoch_mode": str(epoch_mode),
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
        _save_state(out_root, epoch_mode, state)

    for cstate in state["configs"]:
        if cstate.get("done"):
            continue
        cfg_idx = int(cstate["cfg_idx"])
        cfg = dict(cstate["hparams"])
        cfg["cfg_idx"] = cfg_idx
        run_dir = run_dir_for_search_cfg(
            runs_root, cfg_idx, search_train_seed, search_profile
        )

        if cstate.get("done") and (run_dir / "metrics.json").is_file():
            continue

        success_rate, val_loss, rc, epoch_trained = run_search_trial(
            cfg=cfg,
            run_dir=run_dir,
            py=py,
            train_py=train_py,
            infer_py=infer_py,
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
            results_csv=results_csv,
            hp_cols=hp_cols,
            fold_i=fold_i,
            n_infer_episodes=n_infer_episodes,
            n_train_val_episodes=n_train_val_episodes,
            train_val_eval_seed_offset=train_val_eval_seed_offset,
            skip_inference_videos=skip_inference_videos,
        )
        if rc == 0:
            cstate["epoch_trained"] = int(epoch_trained)
        cstate["done"] = True
        state["evaluations"].append(
            {
                "cfg_idx": cfg_idx,
                "test_success_rate_total": success_rate,
                "val_loss": val_loss,
                "returncode": rc,
            }
        )
        _save_state(out_root, epoch_mode, state)
        print(
            f"[random_search:{epoch_mode}] cfg_idx={cfg_idx} selesai "
            f"success_rate={success_rate} val_loss={val_loss} rc={rc}"
        )

    best = _pick_best_from_state(state)
    if best is not None:
        state["best"] = best
        _save_state(out_root, epoch_mode, state)
        print(
            f"[random_search:{epoch_mode}] pemenang cfg_idx={best['cfg_idx']} "
            f"success_rate={best['test_success_rate_total']:.4f}"
        )
    return best
