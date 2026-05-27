"""Konstanta ruang pencarian hyperparameter untuk eksperimen FlowPolicy Kitchen.

Pencarian hiperparameter memakai **random search berpusat di baseline**
(bukan Hyperband). ``training.num_epochs`` ikut disampling dengan bobot
lebih tinggi di sekitar nilai baseline (default 3000).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


def fmt_hydra_val(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return repr(float(v))
    return str(v)


KITCHEN_NUM_POINTS = 512

# Early stop: success total, level k1–k4, dan per sub-tugas Kitchen sequential.
EARLY_STOP_MONITOR_KEYS: List[str] = [
    "success_rate_total",
    "success_rate_k1",
    "success_rate_k2",
    "success_rate_k3",
    "success_rate_k4",
    "success_rate_task_microwave",
    "success_rate_task_kettle",
    "success_rate_task_light_switch",
    "success_rate_task_slide_cabinet",
]

# Plafon batch size random search (baseline tetap pakai --max-batch-size default 128).
SEARCH_DEFAULT_MAX_BATCH_SIZE = 512


def append_kitchen_policy_hparam_overrides(
    odl: List[str],
    cfg: Dict[str, Any],
) -> None:
    """Override Hydra hiperparameter pencarian untuk Franka Kitchen point-cloud.

    Point cloud sudah aktif via ``shape_meta`` + ``KitchenRunner`` /
    ``FrankaKitchenPointCloudEnv`` (default 512 pts). Jangan override key
    yang tidak ada di struct Hydra (mis. ``policy.obs_encoder_type``).
    ``_state_mlp_hidden`` → ``policy.encoder_output_dim``.
    """
    for k in CSV_HPARAM_KEYS:
        if k in ("cfg_idx", "training.num_epochs"):
            continue
        if k == "_state_mlp_hidden":
            odl.append(f"policy.encoder_output_dim={int(cfg[k])}")
            continue
        odl.append(f"{k}={fmt_hydra_val(cfg[k])}")

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
# Pemenang final random search (legacy; rerun dihapus dari pipeline baru).
HYPERBAND_BEST_CFG_IDX = -3
SEARCH_BEST_CFG_IDX = HYPERBAND_BEST_CFG_IDX
SEARCH_CFG_IDX_BASE = 1000
SEARCH_CFG_IDX_BASE_EPOCH_3000 = 2000
HYPERBAND_CFG_IDX_BASE = SEARCH_CFG_IDX_BASE

# Profil baseline (tanpa augmentasi noise).
DEFAULT_PREPROCESSING_PROFILE = "minimal"
# Profil random search (standard = noise observasi std=0.01 pada train).
DEFAULT_SEARCH_PREPROCESSING_PROFILE = "standard"
# Seed training random search (tanpa perlu baseline sebelumnya).
DEFAULT_SEARCH_TRAIN_SEED = 101

# Mode random search: satu epoch tetap per fase (hanya 5000 atau 3000).
EPOCH_SEARCH_MODES: Dict[str, Dict[str, Any]] = {
    "epoch_5000": {"choices": [5000], "center": 5000},
    "epoch_3000": {"choices": [3000], "center": 3000},
}

EPOCH_SEARCH_STATE_FILES: Dict[str, str] = {
    "epoch_5000": "random_search_state_epoch5000.json",
    "epoch_3000": "random_search_state_epoch3000.json",
}

EPOCH_SEARCH_CFG_IDX_BASE: Dict[str, int] = {
    "epoch_5000": SEARCH_CFG_IDX_BASE,
    "epoch_3000": SEARCH_CFG_IDX_BASE_EPOCH_3000,
}

# Ruang pencarian lokal random search — setiap list HARUS mencakup nilai baseline.
LOCAL_SEARCH_SPACE: Dict[str, List[Any]] = {
    "training.num_epochs": [3000, 5000],
    "optimizer.lr": [5e-5, 1e-4, 2e-4, 5e-4],
    "dataloader.batch_size": [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3],
    "policy.Conditional_ConsistencyFM.eps": [1e-3, 1e-2, 5e-2],
    "policy.Conditional_ConsistencyFM.delta": [1e-3, 1e-2, 5e-2],
    "n_action_steps": [2, 4, 6],
    "n_obs_steps": [2, 4, 6],
    "policy.diffusion_step_embed_dim": [128, 256, 512],
    "_state_mlp_hidden": [64, 128, 256],
}

# Ruang Hyperband legacy (tanpa ``training.num_epochs`` — itu resource R).
SEARCH_SPACE = {
    "optimizer.lr": [1e-3, 5e-4, 1e-4, 1e-5],
    "dataloader.batch_size": [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3, 4],
    "policy.Conditional_ConsistencyFM.eps": [1e-4, 1e-3, 1e-2, 0.5],
    "policy.Conditional_ConsistencyFM.delta": [1e-4, 1e-3, 1e-2, 1.0],
    "n_action_steps": [2, 4, 6, 8],
    "n_obs_steps": [2, 4, 6, 8, 16],
    "policy.diffusion_step_embed_dim": [128, 256, 512, 1024],
    "_state_mlp_hidden": [64, 128, 256, 512, 1024],
}

HYPERBAND_SAMPLING_BASELINE_ANCHORED = "baseline_anchored"
HYPERBAND_SAMPLING_RANDOM = "random"
HYPERBAND_DEFAULT_MAX_EPOCHS = 3000
HYPERBAND_DEFAULT_FACTOR = 3
HYPERBAND_DEFAULT_ITERATIONS = 1

CSV_HPARAM_KEYS: List[str] = list(LOCAL_SEARCH_SPACE.keys())


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
    epoch_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Sample ``n`` konfigurasi random search berpusat di ``center`` (default baseline).

    Jika ``epoch_mode`` diset (``epoch_5000`` / ``epoch_3000``), sampling
    ``training.num_epochs`` dibatasi ke pilihan mode tersebut (Gaussian di indeks).
    """
    center = center or DEFAULT_BASELINE_HPARAMS
    space = search_space or LOCAL_SEARCH_SPACE
    epoch_spec: Optional[Dict[str, Any]] = None
    if epoch_mode is not None:
        if epoch_mode not in EPOCH_SEARCH_MODES:
            raise ValueError(
                f"epoch_mode tidak dikenal: {epoch_mode!r} "
                f"(gunakan {list(EPOCH_SEARCH_MODES.keys())!r})"
            )
        epoch_spec = EPOCH_SEARCH_MODES[epoch_mode]

    out: List[Dict[str, Any]] = []
    keys = list(space.keys())

    for i in range(int(n)):
        d: Dict[str, Any] = {"cfg_idx": int(base_cfg_idx) + i}
        use_exact = rng.rand() < float(p_exact_baseline)
        for k in keys:
            if k == "training.num_epochs" and epoch_spec is not None:
                epoch_choices = epoch_spec["choices"]
                epoch_center = epoch_spec["center"]
                if use_exact:
                    d[k] = int(epoch_center)
                else:
                    d[k] = _weighted_choice(
                        rng, epoch_choices, epoch_center, sigma=sigma
                    )
            elif use_exact and k in center:
                d[k] = center[k]
            else:
                d[k] = _weighted_choice(
                    rng, space[k], center.get(k, space[k][0]), sigma=sigma
                )
        out.append(d)
    return out


def baseline_search_center() -> Dict[str, Any]:
    """Pusat pencarian Hyperband = hiperparameter baseline (tanpa epoch/cfg_idx)."""
    return {k: DEFAULT_BASELINE_HPARAMS[k] for k in SEARCH_SPACE.keys()}


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return a == b


def _choice_index(choices: List[Any], value: Any) -> int:
    for i, c in enumerate(choices):
        if _values_equal(c, value):
            return i
    raise ValueError(f"nilai baseline {value!r} tidak ada di pilihan {choices!r}")


def _local_neighbor_choice(
    rng: np.random.RandomState, choices: List[Any], current: Any
) -> Any:
    """Pilih nilai tetangga diskrit ±1 dari ``current`` dalam ``choices``."""
    idx = _choice_index(choices, current)
    lo = max(0, idx - 1)
    hi = min(len(choices) - 1, idx + 1)
    return choices[int(rng.randint(lo, hi + 1))]


def _config_from_center(
    center: Dict[str, Any],
    *,
    cfg_idx: int,
    rng: np.random.RandomState,
    tweak_dims: int,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "cfg_idx": int(cfg_idx),
        "training.num_epochs": 0,
        **{k: center[k] for k in SEARCH_SPACE.keys()},
    }
    if tweak_dims <= 0:
        return d
    keys = list(SEARCH_SPACE.keys())
    n_tweak = min(int(tweak_dims), len(keys))
    for k in rng.choice(keys, size=n_tweak, replace=False):
        d[k] = _local_neighbor_choice(rng, SEARCH_SPACE[k], center[k])
    return d


def sample_configs_hyperband(
    rng: np.random.RandomState,
    n: int,
    *,
    base_cfg_idx: int = HYPERBAND_CFG_IDX_BASE,
    sampling: str = HYPERBAND_SAMPLING_BASELINE_ANCHORED,
    max_dims_to_tweak: int = 4,
) -> List[Dict[str, Any]]:
    """Sample ``n`` konfigurasi untuk Hyperband (modul legacy ``hyperband_search.py``)."""
    n = int(n)
    if n <= 0:
        return []

    mode = str(sampling).lower()
    if mode == HYPERBAND_SAMPLING_RANDOM:
        out: List[Dict[str, Any]] = []
        keys = list(SEARCH_SPACE.keys())
        for i in range(n):
            d = {
                "cfg_idx": int(base_cfg_idx) + i,
                "training.num_epochs": 0,
            }
            for k in keys:
                choices = SEARCH_SPACE[k]
                d[k] = choices[int(rng.randint(0, len(choices)))]
            out.append(d)
        return out

    if mode != HYPERBAND_SAMPLING_BASELINE_ANCHORED:
        raise ValueError(
            f"sampling tidak dikenal: {sampling!r} "
            f"(gunakan {HYPERBAND_SAMPLING_BASELINE_ANCHORED!r} atau "
            f"{HYPERBAND_SAMPLING_RANDOM!r})"
        )

    center = baseline_search_center()
    for k, v in center.items():
        _choice_index(SEARCH_SPACE[k], v)

    out: List[Dict[str, Any]] = []
    for i in range(n):
        if i == 0:
            out.append(
                _config_from_center(
                    center, cfg_idx=base_cfg_idx + i, rng=rng, tweak_dims=0
                )
            )
        else:
            n_dims = int(rng.randint(1, min(max_dims_to_tweak, len(SEARCH_SPACE)) + 1))
            out.append(
                _config_from_center(
                    center,
                    cfg_idx=base_cfg_idx + i,
                    rng=rng,
                    tweak_dims=n_dims,
                )
            )
    return out


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
