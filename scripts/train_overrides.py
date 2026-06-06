#!/usr/bin/env python3
"""Hydra override train — dipakai baseline dan random search."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List

from experiment_constants import (
    EARLY_STOP_MONITOR_KEYS,
    append_kitchen_policy_hparam_overrides,
    compute_horizon,
)

_DEBUG_LOG = "/home/daffa/Documents/kripsy12/.cursor/debug-8a2c7a.log"


def _dbg_log(location: str, message: str, data: dict, hypothesis_id: str = "A") -> None:
    # #region agent log
    import json
    import time

    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "8a2c7a",
                        "location": location,
                        "message": message,
                        "data": data,
                        "hypothesisId": hypothesis_id,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion


def build_train_overrides(
    cfg: Dict[str, Any],
    *,
    seed: int,
    profile: str,
    train_eps: List[int],
    val_eps: List[int],
    run_dir: pathlib.Path,
    zarr_rel: str,
    resume_training: bool,
    checkpoint_every: int,
    dataloader_num_workers: int,
    enable_early_stop: bool = True,
    early_stop_rollout_every: int = 200,
) -> List[str]:
    n_obs = int(cfg["n_obs_steps"])
    n_act = int(cfg["n_action_steps"])
    hz = compute_horizon(n_obs, n_act)
    bs = int(cfg["dataloader.batch_size"])

    def il(xs: List[int]) -> str:
        return "[" + ",".join(str(int(x)) for x in xs) + "]"

    odl: List[str] = [
        "task=franka_kitchen_complete4",
        f"task.dataset.zarr_path={zarr_rel}",
        f"task.dataset.train_episode_indices={il(train_eps)}",
        f"task.dataset.val_episode_indices={il(val_eps)}",
        f"task.dataset.preprocessing_profile={profile}",
        f"training.seed={seed}",
        f"task.dataset.seed={seed}",
        "training.compute_val_loss=true",
        f"training.resume={str(resume_training).lower()}",
        "checkpoint.save_ckpt=true",
        f"training.checkpoint_every={checkpoint_every}",
        "checkpoint.save_last_ckpt=true",
        "logging.mode=offline",
        f"hydra.run.dir={run_dir.resolve()}",
        "hydra.job.chdir=true",
        f"horizon={hz}",
        f"n_obs_steps={n_obs}",
        f"n_action_steps={n_act}",
        f"dataloader.batch_size={bs}",
        f"val_dataloader.batch_size={bs}",
        f"dataloader.num_workers={dataloader_num_workers}",
        f"val_dataloader.num_workers={dataloader_num_workers}",
    ]
    if enable_early_stop:
        mk_hydra = "[" + ",".join(EARLY_STOP_MONITOR_KEYS) + "]"
        odl.extend(
            [
                "training.early_stop.enabled=true",
                f"training.rollout_every={int(early_stop_rollout_every)}",
                f"training.early_stop.monitor_keys={mk_hydra}",
            ]
        )
    else:
        odl.append("training.rollout_every=999999")

    odl.append(f"training.num_epochs={int(cfg['training.num_epochs'])}")
    append_kitchen_policy_hparam_overrides(odl, cfg)
    _dbg_log(
        "train_overrides.py:build_train_overrides",
        "hydra overrides built",
        {
            "enable_early_stop": enable_early_stop,
            "early_stop_in_odl": any("early_stop" in x for x in odl),
            "invalid_policy_keys": [
                x for x in odl if "obs_encoder_type" in x or "obs_mode" in x
            ],
            "rollout_every": early_stop_rollout_every if enable_early_stop else 999999,
            "num_epochs": int(cfg["training.num_epochs"]),
            "profile": profile,
            "seed": seed,
        },
        hypothesis_id="D",
    )
    return odl
