"""Konstanta ruang pencarian hyperparameter untuk eksperimen FlowPolicy Kitchen.

Pencarian hiperparameter memakai Hyperband (Li et al., 2018,
https://arxiv.org/pdf/1603.06560). Karena ``training.num_epochs`` adalah
resource Hyperband (R), kunci ini DIKELUARKAN dari ``SEARCH_SPACE`` agar tidak
disampling sebagai dimensi pencarian. Nilai ``training.num_epochs`` yang
benar-benar dilatih untuk tiap baris ``results.csv`` tetap dicatat sebagai
kolom hiperparameter (``CSV_HPARAM_KEYS``) — untuk baseline = 3000,
untuk pemenang Hyperband final = R.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def fmt_hydra_val(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return repr(float(v))
    return str(v)


def append_kitchen_policy_hparam_overrides(
    odl: List[str],
    cfg: Dict[str, Any],
) -> None:
    """Override Hydra untuk Franka Kitchen: state encoder + ruang ``SEARCH_SPACE``.

    Selalu memaksa ``obs_encoder_type=state`` (59-dim, tanpa PointNet) seperti
    ``franka_kitchen_complete4``. ``_state_mlp_hidden`` → ``encoder_output_dim``;
    hidden MLP state encoder mengikuti ``state_encoder_cfg`` task (default [256,256]).
    """
    odl.append("policy.obs_encoder_type=state")
    odl.append("task.env_runner.obs_mode=state")
    for k in CSV_HPARAM_KEYS:
        if k in ("cfg_idx", "training.num_epochs"):
            continue
        if k == "_state_mlp_hidden":
            # Ukuran keluaran StateFlowPolicyEncoder; hidden MLP dari task YAML [256,256].
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
# Pemenang final Hyperband yang di-rerun pada 3 seeds × 2 profiles.
HYPERBAND_BEST_CFG_IDX = -3
# Cfg_idx untuk konfigurasi yang dievaluasi di dalam fase Hyperband
# (mulai dari basis ini agar tidak bentrok dengan baseline / pemenang final).
HYPERBAND_CFG_IDX_BASE = 1000

# Ruang pencarian Hyperband (tanpa ``training.num_epochs`` — itu resource R).
# Nilai baseline (``DEFAULT_BASELINE_HPARAMS``) harus ada di setiap daftar agar
# tweak lokal di sekitar konfigurasi terbukti Franka Kitchen bisa dilakukan.
SEARCH_SPACE = {
    "optimizer.lr": [1e-3, 5e-4, 1e-4, 1e-5],
    "dataloader.batch_size": [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3, 4],
    "policy.Conditional_ConsistencyFM.eps": [1e-4, 1e-3, 1e-2, 0.5],
    "policy.Conditional_ConsistencyFM.delta": [1e-4, 1e-3, 1e-2, 1.0],
    "n_action_steps": [2, 4, 6, 8],
    "n_obs_steps": [2, 4, 6, 8, 16],
    "policy.diffusion_step_embed_dim": [128, 256, 512, 1024],
    # Ukuran keluaran StateFlowPolicyEncoder (``policy.encoder_output_dim``).
    "_state_mlp_hidden": [64, 128, 256, 512, 1024],
}

# Mode sampling Hyperband: ``baseline_anchored`` = warm-start dari baseline terbukti.
HYPERBAND_SAMPLING_BASELINE_ANCHORED = "baseline_anchored"
HYPERBAND_SAMPLING_RANDOM = "random"


# Alias selaras API KerasTuner Hyperband (https://keras.io/keras_tuner/api/tuners/hyperband/)
# — dipakai di CLI ``run_experiment.py`` / ``hyperband_search.py``.
HYPERBAND_DEFAULT_MAX_EPOCHS = 3000  # max_epochs
HYPERBAND_DEFAULT_FACTOR = 3  # factor (eta)
HYPERBAND_DEFAULT_ITERATIONS = 1  # hyperband_iterations

# Kolom hiperparameter di CSV (tanpa prefix policy untuk CFM agar rapi).
# Kolom pertama: ``training.num_epochs`` — nilai aktual epoch yang dilatih
# (baseline = 3000; Hyperband final winner = R; baris intermediate HB = r_i terakhir).
CSV_HPARAM_KEYS: List[str] = ["training.num_epochs"] + list(SEARCH_SPACE.keys())


def compute_horizon(n_obs_steps: int, n_action_steps: int) -> int:
    return 4 * ((max(n_obs_steps + n_action_steps - 1, 4) + 3) // 4)


def baseline_config_dict() -> dict:
    """Salinan baseline dengan cfg_idx untuk CSV dan orchestrator."""
    out = dict(DEFAULT_BASELINE_HPARAMS)
    out["cfg_idx"] = BASELINE_CFG_IDX
    return out


def baseline_search_center() -> Dict[str, Any]:
    """Pusat pencarian = hiperparameter baseline Franka Kitchen (tanpa epoch/cfg_idx)."""
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
    """Pilih nilai tetangga diskrit ±1 dari ``current`` dalam ``choices`` (tweak lokal)."""
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
    """Salin ``center`` dan ubah ``tweak_dims`` dimensi ke tetangga lokal di ``SEARCH_SPACE``."""
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
    """Sample ``n`` konfigurasi untuk Hyperband.

    ``training.num_epochs`` TIDAK disampling (= resource Hyperband).

    **baseline_anchored** (default): warm-start dari ``DEFAULT_BASELINE_HPARAMS``
    (konfigurasi terbukti di Franka Kitchen, selaras
    ``flowpolicy_hyperparameter_finetuning.md`` §1–2):

    - Trial pertama di setiap bracket: **baseline persis**
    - Trial lain: tweak lokal (ubah 1–``max_dims_to_tweak`` dimensi ke nilai
      tetangga dalam ``SEARCH_SPACE``, bukan cold random di seluruh ruang)

    **random**: uniform cold start (legacy).
    """
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
    # Pastikan baseline valid terhadap SEARCH_SPACE (fail-fast).
    for k, v in center.items():
        _choice_index(SEARCH_SPACE[k], v)

    out: List[Dict[str, Any]] = []
    for i in range(n):
        if i == 0:
            out.append(_config_from_center(center, cfg_idx=base_cfg_idx + i, rng=rng, tweak_dims=0))
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
