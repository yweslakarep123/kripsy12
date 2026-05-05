"""Konstanta ruang pencarian hyperparameter untuk eksperimen FlowPolicy Kitchen."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# Selaras dengan `flowpolicy.yaml` + `franka_kitchen_complete4` (FlowPolicy asli).
DEFAULT_BASELINE_HPARAMS = {
    "training.num_epochs": 3000,
    "optimizer.lr": 1e-4,
    "dataloader.batch_size": 128,
    "policy.Conditional_ConsistencyFM.num_segments": 2,
    "policy.Conditional_ConsistencyFM.eps": 1e-2,
    "policy.Conditional_ConsistencyFM.delta": 1e-2,
    "n_action_steps": 4,
    "n_obs_steps": 2,
    "policy.diffusion_step_embed_dim": 128,
    "_state_mlp_hidden": 64,
}

BASELINE_CFG_IDX = -1

SEARCH_SPACE = {
    "training.num_epochs": [500, 1000, 3000, 5000],
    "optimizer.lr": [1e-3, 5e-4, 1e-4, 1e-5],
    "dataloader.batch_size": [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3, 4],
    "policy.Conditional_ConsistencyFM.eps": [1e-4, 1e-3, 1e-2, 1.0],
    "policy.Conditional_ConsistencyFM.delta": [1e-4, 1e-3, 1e-2, 1.0],
    "n_action_steps": [2, 4, 6, 8],
    "n_obs_steps": [4, 6, 8, 16],
    "policy.diffusion_step_embed_dim": [128, 256, 512, 1024],
    "_state_mlp_hidden": [128, 256, 512, 1024],
}

# Kolom hiperparameter di CSV (tanpa prefix policy untuk CFM agar rapi)
CSV_HPARAM_KEYS = list(SEARCH_SPACE.keys())


def compute_horizon(n_obs_steps: int, n_action_steps: int) -> int:
    return 4 * ((max(n_obs_steps + n_action_steps - 1, 4) + 3) // 4)


def baseline_config_dict() -> dict:
    """Salinan baseline dengan cfg_idx untuk CSV dan orchestrator."""
    out = dict(DEFAULT_BASELINE_HPARAMS)
    out["cfg_idx"] = BASELINE_CFG_IDX
    return out


def sample_configs(rng: np.random.RandomState, n: int) -> List[Dict[str, Any]]:
    """Random search: `cfg_idx` 0 .. n-1."""
    out: List[Dict[str, Any]] = []
    keys = list(SEARCH_SPACE.keys())
    for i in range(n):
        d: Dict[str, Any] = {"cfg_idx": i}
        for k in keys:
            choices = SEARCH_SPACE[k]
            d[k] = choices[int(rng.randint(0, len(choices)))]
        out.append(d)
    return out
