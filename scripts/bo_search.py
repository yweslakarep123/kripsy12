"""Optimasi Bayesian (GP + expected improvement) di atas ruang kategorikal SEARCH_SPACE."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    from skopt import Optimizer, dump, load
    from skopt.space import Categorical
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Butuh paket scikit-optimize. Contoh: pip install scikit-optimize"
    ) from e

from experiment_constants import (
    CSV_HPARAM_KEYS,
    SEARCH_SPACE,
    config_dict_to_vector,
)


def skopt_dimensions() -> List[Categorical]:
    return [
        Categorical(categories=list(SEARCH_SPACE[k]), name=str(k))
        for k in CSV_HPARAM_KEYS
    ]


def bo_optimizer_path(out_root: pathlib.Path) -> pathlib.Path:
    return out_root / "bo_optimizer.pkl"


def create_optimizer(
    *,
    random_state: int,
    n_initial_points: int,
) -> Optimizer:
    return Optimizer(
        skopt_dimensions(),
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="sampling",
        random_state=np.random.RandomState(int(random_state)),
        n_initial_points=int(n_initial_points),
    )


def load_or_create_optimizer(
    out_root: pathlib.Path,
    *,
    random_state: int,
    n_initial_points: int,
) -> Optimizer:
    p = bo_optimizer_path(out_root)
    if p.is_file():
        return load(str(p))
    return create_optimizer(
        random_state=random_state, n_initial_points=n_initial_points
    )


def ensure_optimizer_matches_sampled(
    out_root: pathlib.Path,
    sampled_cfgs: List[Dict[str, Any]],
    results_csv: pathlib.Path,
    seeds: Sequence[int],
    profiles: Sequence[str],
    *,
    random_state: int,
    n_initial_points: int,
    objective_mode: str,
) -> Optimizer:
    """
    Muat ``bo_optimizer.pkl`` jika jumlah observasi = len(sampled_cfgs);
    jika tidak, bangun ulang dari ``results.csv`` + ``sampled_cfgs``.
    """
    p = bo_optimizer_path(out_root)
    if p.is_file():
        try:
            opt = load(str(p))
            if len(opt.Xi) == len(sampled_cfgs):
                return opt
        except Exception:
            pass

    opt = create_optimizer(
        random_state=random_state, n_initial_points=n_initial_points
    )
    for cfg in sampled_cfgs:
        xi = config_dict_to_vector(cfg)
        yi = aggregate_objective_from_results_csv(
            results_csv,
            int(cfg["cfg_idx"]),
            seeds,
            profiles,
            mode=objective_mode,
        )
        tell_objective(opt, xi, yi)
    save_optimizer(opt, out_root)
    return opt


def save_optimizer(opt: Optimizer, out_root: pathlib.Path) -> None:
    dump(opt, str(bo_optimizer_path(out_root)))


def ask_next_config(opt: Optimizer) -> List[Any]:
    """Satu titik berikutnya (urutan = CSV_HPARAM_KEYS)."""
    pts = opt.ask(n_points=1, strategy="cl_min")
    return list(pts[0])


def tell_objective(opt: Optimizer, x: Sequence[Any], y_minimize: float) -> None:
    opt.tell(list(x), float(y_minimize))


def aggregate_objective_from_results_csv(
    results_csv: pathlib.Path,
    cfg_idx: int,
    seeds: Sequence[int],
    profiles: Sequence[str],
    *,
    mode: str = "neg_test_trade_off",
) -> float:
    """
    Agregasi lintas seed×profile untuk satu cfg_idx (hanya status=ok).
    BO meminimalkan nilai return: default ``-mean(test_trade_off)``.
    """
    if not results_csv.is_file():
        return 1e9
    df = pd.read_csv(results_csv)
    if df.empty:
        return 1e9
    sub = df[
        (df["cfg_idx"].astype(int) == int(cfg_idx))
        & (df["status"].astype(str) == "ok")
        & (df["seed"].isin(list(seeds)))
        & (df["profile"].astype(str).isin(list(profiles)))
    ]
    if sub.empty:
        return 1e9

    if "test_trade_off" in sub.columns:
        to = pd.to_numeric(sub["test_trade_off"], errors="coerce")
    else:
        to = pd.to_numeric(sub["trade_off"], errors="coerce")

    mean_to = float(to.mean(skipna=True))
    if np.isnan(mean_to):
        return 1e9

    if mode == "neg_test_trade_off":
        return float(-mean_to)
    if mode == "neg_test_success_rate_k4":
        col = "test_success_rate_k4" if "test_success_rate_k4" in sub.columns else "success_rate_k4"
        v = float(pd.to_numeric(sub[col], errors="coerce").mean(skipna=True))
        if np.isnan(v):
            return 1e9
        return float(-v)
    raise ValueError(f"mode tidak dikenal: {mode}")
