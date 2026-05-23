"""Konstanta ruang pencarian hyperparameter untuk eksperimen FlowPolicy Kitchen.

Pencarian hiperparameter memakai **random search berpusat di baseline**
(bukan Hyperband). ``training.num_epochs`` ikut disampling dengan bobot
lebih tinggi di sekitar nilai baseline (default 3000).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

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
# Pemenang final random search yang di-rerun pada 3 seeds × 2 profiles.
HYPERBAND_BEST_CFG_IDX = -3
SEARCH_BEST_CFG_IDX = HYPERBAND_BEST_CFG_IDX
# Cfg_idx untuk konfigurasi yang dievaluasi di dalam fase pencarian.
SEARCH_CFG_IDX_BASE = 1000
HYPERBAND_CFG_IDX_BASE = SEARCH_CFG_IDX_BASE

# Ruang pencarian lokal — setiap list HARUS mencakup nilai baseline.
LOCAL_SEARCH_SPACE: Dict[str, List[Any]] = {
    "training.num_epochs": [1000, 2000, 2500, 3000, 3500, 4000, 5000],
    "optimizer.lr": [5e-5, 1e-4, 2e-4, 5e-4],
    "dataloader.batch_size": [64, 128, 256],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3],
    "policy.Conditional_ConsistencyFM.eps": [1e-3, 1e-2, 5e-2],
    "policy.Conditional_ConsistencyFM.delta": [1e-3, 1e-2, 5e-2],
    "n_action_steps": [2, 4, 6],
    "n_obs_steps": [2, 4, 6],
    "policy.diffusion_step_embed_dim": [128, 256, 512],
    "_state_mlp_hidden": [64, 128, 256],
}

# Alias kompatibilitas lama.
SEARCH_SPACE = LOCAL_SEARCH_SPACE

# Kolom hiperparameter di CSV.
CSV_HPARAM_KEYS: List[str] = ["training.num_epochs"] + [
    k for k in LOCAL_SEARCH_SPACE.keys()
]


def compute_horizon(n_obs_steps: int, n_action_steps: int) -> int:
    return 4 * ((max(n_obs_steps + n_action_steps - 1, 4) + 3) // 4)


def baseline_config_dict() -> dict:
    """Salinan baseline dengan cfg_idx untuk CSV dan orchestrator."""
    out = dict(DEFAULT_BASELINE_HPARAMS)
    out["cfg_idx"] = BASELINE_CFG_IDX
    return out


def _center_index(choices: List[Any], center_value: Any) -> int:
    """Indeks pilihan terdekat ke ``center_value`` (log-scale untuk angka positif)."""

    def dist(a: Any, b: Any) -> float:
        try:
            fa, fb = float(a), float(b)
            if fa > 0 and fb > 0:
                return abs(math.log(fa) - math.log(fb))
            return abs(fa - fb)
        except (TypeError, ValueError):
            return 0.0 if a == b else 1.0

    return min(range(len(choices)), key=lambda i: dist(choices[i], center_value))


def _weighted_choice(
    rng: np.random.RandomState,
    choices: List[Any],
    center_value: Any,
    *,
    sigma: float = 1.0,
) -> Any:
    """Sample dari ``choices`` dengan bobot Gaussian pada jarak indeks dari baseline."""
    if not choices:
        raise ValueError("choices kosong")
    if len(choices) == 1:
        return choices[0]
    cidx = _center_index(choices, center_value)
    weights = np.array(
        [math.exp(-0.5 * ((i - cidx) / sigma) ** 2) for i in range(len(choices))],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return choices[int(rng.choice(len(choices), p=weights))]


def sample_configs_around_baseline(
    rng: np.random.RandomState,
    n: int,
    *,
    center: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, List[Any]]] = None,
    base_cfg_idx: int = SEARCH_CFG_IDX_BASE,
    sigma: float = 1.0,
    p_exact_baseline: float = 0.15,
) -> List[Dict[str, Any]]:
    """Sample ``n`` konfigurasi random search berpusat di ``center`` (default baseline).

    Nilai dekat baseline (mis. epoch 3000 → 2500/3500 lebih sering daripada 1000/5000)
    dipilih via bobot Gaussian pada jarak indeks di ``LOCAL_SEARCH_SPACE``.
    """
    center = center or DEFAULT_BASELINE_HPARAMS
    space = search_space or LOCAL_SEARCH_SPACE
    out: List[Dict[str, Any]] = []
    keys = list(space.keys())

    for i in range(int(n)):
        d: Dict[str, Any] = {"cfg_idx": int(base_cfg_idx) + i}
        use_exact = rng.rand() < float(p_exact_baseline)
        for k in keys:
            if use_exact and k in center:
                d[k] = center[k]
            else:
                d[k] = _weighted_choice(
                    rng, space[k], center.get(k, space[k][0]), sigma=sigma
                )
        out.append(d)
    return out


def sample_configs_hyperband(
    rng: np.random.RandomState,
    n: int,
    *,
    base_cfg_idx: int = SEARCH_CFG_IDX_BASE,
) -> List[Dict[str, Any]]:
    """Alias lama — delegasi ke ``sample_configs_around_baseline`` (uniform sigma)."""
    return sample_configs_around_baseline(
        rng, n, base_cfg_idx=base_cfg_idx, sigma=1.5, p_exact_baseline=0.0
    )


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
