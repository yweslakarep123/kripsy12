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
    "policy.Conditional_ConsistencyFM.eps": [1e-4, 1e-3, 1e-2, 0.5],
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


def config_vector_to_dict(x: List[Any], cfg_idx: int) -> Dict[str, Any]:
    """Satu titik ruang pencarian (urutan = CSV_HPARAM_KEYS) → dict cfg Hydra."""
    d: Dict[str, Any] = {"cfg_idx": int(cfg_idx)}
    for k, v in zip(CSV_HPARAM_KEYS, x):
        d[k] = v
    return d


def config_dict_to_vector(cfg: Dict[str, Any]) -> List[Any]:
    return [cfg[k] for k in CSV_HPARAM_KEYS]


# Kolom tambahan results.csv (metrik infer dua fase + alias kompatibel).
RESULTS_CSV_METRIC_COLUMNS = [
    "training_sim_success_rate_total",
    "training_sim_success_rate_k1",
    "training_sim_success_rate_k2",
    "training_sim_success_rate_k3",
    "training_sim_success_rate_k4",
    "training_sim_mean_inference_latency_ms",
    "training_sim_std_inference_latency_ms",
    "training_sim_mean_episode_mean_inference_latency_ms",
    "training_sim_std_episode_mean_inference_latency_ms",
    "training_sim_trade_off",
    "training_sim_trade_off_episode_latency",
    "training_sim_n_infer_episodes",
    "train_val_success_rate_total",
    "train_val_success_rate_k1",
    "train_val_success_rate_k2",
    "train_val_success_rate_k3",
    "train_val_success_rate_k4",
    "train_val_mean_inference_latency_ms",
    "train_val_std_inference_latency_ms",
    "train_val_mean_episode_mean_inference_latency_ms",
    "train_val_std_episode_mean_inference_latency_ms",
    "train_val_trade_off",
    "train_val_trade_off_episode_latency",
    "train_val_n_infer_episodes",
    "test_success_rate_total",
    "test_success_rate_k1",
    "test_success_rate_k2",
    "test_success_rate_k3",
    "test_success_rate_k4",
    "test_mean_inference_latency_ms",
    "test_std_inference_latency_ms",
    "test_mean_episode_mean_inference_latency_ms",
    "test_std_episode_mean_inference_latency_ms",
    "test_trade_off",
    "test_trade_off_episode_latency",
    "test_n_infer_episodes",
    "success_rate_total",
    "success_rate_k1",
    "success_rate_k2",
    "success_rate_k3",
    "success_rate_k4",
    "mean_inference_latency_ms",
    "std_inference_latency_ms",
    "mean_episode_mean_inference_latency_ms",
    "std_episode_mean_inference_latency_ms",
    "trade_off",
    "trade_off_episode_latency",
]


def metrics_row_from_infer_json(met: Dict[str, Any]) -> Dict[str, Any]:
    """Isi kolom metrik CSV dari metrics.json (format baru bertahap atau legacy)."""

    def pick(*names: str, default: Any = "") -> Any:
        for n in names:
            if n in met and met[n] is not None:
                return met[n]
        return default

    has_tv = "train_val_success_rate_k1" in met
    has_ts = "training_sim_success_rate_k1" in met

    row: Dict[str, Any] = {}

    if has_ts:
        row["training_sim_success_rate_total"] = pick(
            "training_sim_success_rate_total"
        )
        row["training_sim_success_rate_k1"] = pick("training_sim_success_rate_k1")
        row["training_sim_success_rate_k2"] = pick("training_sim_success_rate_k2")
        row["training_sim_success_rate_k3"] = pick("training_sim_success_rate_k3")
        row["training_sim_success_rate_k4"] = pick("training_sim_success_rate_k4")
        row["training_sim_mean_inference_latency_ms"] = pick(
            "training_sim_mean_inference_latency_ms"
        )
        row["training_sim_std_inference_latency_ms"] = pick(
            "training_sim_std_inference_latency_ms"
        )
        row["training_sim_mean_episode_mean_inference_latency_ms"] = pick(
            "training_sim_mean_episode_mean_inference_latency_ms"
        )
        row["training_sim_std_episode_mean_inference_latency_ms"] = pick(
            "training_sim_std_episode_mean_inference_latency_ms"
        )
        row["training_sim_trade_off"] = pick("training_sim_trade_off")
        row["training_sim_trade_off_episode_latency"] = pick(
            "training_sim_trade_off_episode_latency"
        )
        row["training_sim_n_infer_episodes"] = pick("training_sim_n_infer_episodes")
    else:
        for c in (
            "training_sim_success_rate_total",
            "training_sim_success_rate_k1",
            "training_sim_success_rate_k2",
            "training_sim_success_rate_k3",
            "training_sim_success_rate_k4",
            "training_sim_mean_inference_latency_ms",
            "training_sim_std_inference_latency_ms",
            "training_sim_mean_episode_mean_inference_latency_ms",
            "training_sim_std_episode_mean_inference_latency_ms",
            "training_sim_trade_off",
            "training_sim_trade_off_episode_latency",
            "training_sim_n_infer_episodes",
        ):
            row[c] = ""

    if has_tv:
        row["train_val_success_rate_total"] = pick("train_val_success_rate_total")
        row["train_val_success_rate_k1"] = pick("train_val_success_rate_k1")
        row["train_val_success_rate_k2"] = pick("train_val_success_rate_k2")
        row["train_val_success_rate_k3"] = pick("train_val_success_rate_k3")
        row["train_val_success_rate_k4"] = pick("train_val_success_rate_k4")
        row["train_val_mean_inference_latency_ms"] = pick(
            "train_val_mean_inference_latency_ms"
        )
        row["train_val_std_inference_latency_ms"] = pick(
            "train_val_std_inference_latency_ms"
        )
        row["train_val_mean_episode_mean_inference_latency_ms"] = pick(
            "train_val_mean_episode_mean_inference_latency_ms"
        )
        row["train_val_std_episode_mean_inference_latency_ms"] = pick(
            "train_val_std_episode_mean_inference_latency_ms"
        )
        row["train_val_trade_off"] = pick("train_val_trade_off")
        row["train_val_trade_off_episode_latency"] = pick(
            "train_val_trade_off_episode_latency"
        )
        row["train_val_n_infer_episodes"] = pick("train_val_n_infer_episodes")
    else:
        for c in (
            "train_val_success_rate_total",
            "train_val_success_rate_k1",
            "train_val_success_rate_k2",
            "train_val_success_rate_k3",
            "train_val_success_rate_k4",
            "train_val_mean_inference_latency_ms",
            "train_val_std_inference_latency_ms",
            "train_val_mean_episode_mean_inference_latency_ms",
            "train_val_std_episode_mean_inference_latency_ms",
            "train_val_trade_off",
            "train_val_trade_off_episode_latency",
            "train_val_n_infer_episodes",
        ):
            row[c] = ""

    row["test_success_rate_total"] = pick(
        "test_success_rate_total", "success_rate_total"
    )
    row["test_success_rate_k1"] = pick("test_success_rate_k1", "success_rate_k1")
    row["test_success_rate_k2"] = pick("test_success_rate_k2", "success_rate_k2")
    row["test_success_rate_k3"] = pick("test_success_rate_k3", "success_rate_k3")
    row["test_success_rate_k4"] = pick("test_success_rate_k4", "success_rate_k4")
    row["test_mean_inference_latency_ms"] = pick(
        "test_mean_inference_latency_ms", "mean_inference_latency_ms"
    )
    row["test_std_inference_latency_ms"] = pick(
        "test_std_inference_latency_ms", "std_inference_latency_ms"
    )
    row["test_mean_episode_mean_inference_latency_ms"] = pick(
        "test_mean_episode_mean_inference_latency_ms",
        "mean_episode_mean_inference_latency_ms",
    )
    row["test_std_episode_mean_inference_latency_ms"] = pick(
        "test_std_episode_mean_inference_latency_ms",
        "std_episode_mean_inference_latency_ms",
    )
    row["test_trade_off"] = pick("test_trade_off", "trade_off")
    row["test_trade_off_episode_latency"] = pick(
        "test_trade_off_episode_latency", "trade_off_episode_latency"
    )
    row["test_n_infer_episodes"] = pick(
        "test_n_infer_episodes", "n_infer_episodes"
    )

    row["success_rate_total"] = pick(
        "success_rate_total", "test_success_rate_total"
    )
    row["success_rate_k1"] = pick("success_rate_k1", "test_success_rate_k1")
    row["success_rate_k2"] = pick("success_rate_k2", "test_success_rate_k2")
    row["success_rate_k3"] = pick("success_rate_k3", "test_success_rate_k3")
    row["success_rate_k4"] = pick("success_rate_k4", "test_success_rate_k4")
    row["mean_inference_latency_ms"] = pick(
        "mean_inference_latency_ms", "test_mean_inference_latency_ms"
    )
    row["std_inference_latency_ms"] = pick(
        "std_inference_latency_ms", "test_std_inference_latency_ms"
    )
    row["mean_episode_mean_inference_latency_ms"] = pick(
        "mean_episode_mean_inference_latency_ms",
        "test_mean_episode_mean_inference_latency_ms",
    )
    row["std_episode_mean_inference_latency_ms"] = pick(
        "std_episode_mean_inference_latency_ms",
        "test_std_episode_mean_inference_latency_ms",
    )
    row["trade_off"] = pick("trade_off", "test_trade_off")
    row["trade_off_episode_latency"] = pick(
        "trade_off_episode_latency", "test_trade_off_episode_latency"
    )

    return row


def empty_metrics_row() -> Dict[str, Any]:
    """Nilai kosong untuk semua kolom metrik results.csv."""
    return {k: "" for k in RESULTS_CSV_METRIC_COLUMNS}
