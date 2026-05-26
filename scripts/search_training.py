#!/usr/bin/env python3
"""Jalankan satu trial random search — train + inferensi + results.csv."""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from experiment_constants import (
    CSV_HPARAM_KEYS,
    empty_metrics_row,
    metrics_row_from_infer_json,
)
from infer_utils import run_infer_subprocess
from train_overrides import build_train_overrides


def read_val_loss_final(run_dir: pathlib.Path) -> Optional[float]:
    p = run_dir / "training_final.json"
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return None
    v = d.get("val_loss_final")
    if v is None:
        return None
    try:
        vf = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(vf) or math.isinf(vf):
        return None
    return vf


def read_test_success_rate(run_dir: pathlib.Path) -> Optional[float]:
    p = run_dir / "metrics.json"
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            met = json.load(f)
    except Exception:
        return None
    for key in ("test_success_rate_total", "success_rate_total"):
        v = met.get(key)
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(vf) or math.isinf(vf):
            continue
        return vf
    return None


def run_dir_for_search_cfg(
    runs_root: pathlib.Path, cfg_idx: int, seed: int, profile: str
) -> pathlib.Path:
    return runs_root / f"rs_cfg{int(cfg_idx)}_seed{int(seed)}_{profile}"


def print_trial_configuration(
    label: str,
    cfg: Dict[str, Any],
    overrides: List[str],
    run_dir: pathlib.Path,
) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("Folder run:", run_dir.resolve())
    print("-" * 72)
    print("Hyperparameter (flat):")
    hp = {k: cfg[k] for k in CSV_HPARAM_KEYS if k in cfg}
    print(json.dumps(hp, indent=2, default=str))
    print("-" * 72)
    print("Override Hydra (train):")
    for line in sorted(overrides):
        print(" ", line)
    print("=" * 72 + "\n")


def _load_training_final(run_dir: pathlib.Path) -> Tuple[Any, Any]:
    p = run_dir / "training_final.json"
    if not p.is_file():
        return None, None
    with open(p) as f:
        d = json.load(f)
    return d.get("train_loss_final"), d.get("val_loss_final")


def append_search_results_csv(
    results_csv: pathlib.Path,
    *,
    cfg: Dict[str, Any],
    cfg_idx: int,
    seed: int,
    profile: str,
    fold_i: int,
    hp_cols: List[str],
    run_dir: pathlib.Path,
    ckpt_path: pathlib.Path,
    metrics_path: pathlib.Path,
    n_infer_episodes: int,
    status: str,
) -> None:
    import csv

    from experiment_constants import RESULTS_CSV_METRIC_COLUMNS

    tr_l, va_l = _load_training_final(run_dir)
    row: Dict[str, Any] = {
        "cfg_idx": cfg_idx,
        "seed": seed,
        "profile": profile,
        "fold": fold_i,
        **{k: cfg[k] for k in hp_cols},
        **empty_metrics_row(),
        "train_loss_final": tr_l,
        "val_loss_final": va_l,
        "n_infer_episodes": n_infer_episodes,
        "checkpoint_path": str(ckpt_path),
        "status": status,
    }
    if metrics_path.is_file():
        with open(metrics_path) as f:
            met = json.load(f)
        mrow = metrics_row_from_infer_json(met)
        row.update(mrow)
        row["n_infer_episodes"] = met.get(
            "test_n_infer_episodes",
            met.get("n_infer_episodes", n_infer_episodes),
        )

    fieldnames = (
        ["cfg_idx", "seed", "profile", "fold"]
        + hp_cols
        + list(RESULTS_CSV_METRIC_COLUMNS)
        + [
            "train_loss_final",
            "val_loss_final",
            "n_infer_episodes",
            "checkpoint_path",
            "status",
        ]
    )
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.is_file()
    with open(results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_search_trial(
    *,
    cfg: Dict[str, Any],
    run_dir: pathlib.Path,
    py: str,
    train_py: pathlib.Path,
    infer_py: pathlib.Path,
    cwd_train: str,
    seed: int,
    profile: str,
    train_eps: List[int],
    val_eps: List[int],
    zarr_rel: str,
    checkpoint_every: int,
    dataloader_num_workers: int,
    enable_early_stop: bool = True,
    early_stop_rollout_every: int = 200,
    results_csv: Optional[pathlib.Path] = None,
    hp_cols: Optional[List[str]] = None,
    fold_i: int = 0,
    n_infer_episodes: int = 50,
    n_train_val_episodes: int = 15,
    train_val_eval_seed_offset: int = 31,
    skip_inference_videos: bool = False,
) -> Tuple[Optional[float], Optional[float], int, int]:
    """Latih + inferensi satu konfigurasi search.

    Kembalikan ``(test_success_rate, val_loss, returncode, target_epochs)``.
    """
    run_dir = pathlib.Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    target_epochs = int(cfg["training.num_epochs"])
    cfg_idx = int(cfg["cfg_idx"])
    metrics_path = run_dir / "metrics.json"
    ckpt_path = run_dir / "checkpoints" / "latest.ckpt"

    if metrics_path.is_file():
        v = read_val_loss_final(run_dir)
        sr = read_test_success_rate(run_dir)
        return sr, v, 0, target_epochs

    tf = run_dir / "training_final.json"
    infer_only = tf.is_file() and ckpt_path.is_file() and not metrics_path.is_file()

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")

    if not infer_only:
        ckpt = run_dir / "checkpoints" / "latest.ckpt"
        resume_training = bool(ckpt.is_file() and not tf.is_file())

        overrides = build_train_overrides(
            cfg,
            seed=seed,
            profile=profile,
            train_eps=train_eps,
            val_eps=val_eps,
            run_dir=run_dir,
            zarr_rel=zarr_rel,
            resume_training=resume_training,
            checkpoint_every=checkpoint_every,
            dataloader_num_workers=dataloader_num_workers,
            enable_early_stop=enable_early_stop,
            early_stop_rollout_every=early_stop_rollout_every,
        )

        print_trial_configuration(
            f"[random_search] train cfg_idx={cfg_idx} seed={seed} profile={profile}"
            + (f" (resume={resume_training})" if resume_training else ""),
            cfg,
            overrides,
            run_dir,
        )

        cmd = [py, str(train_py)] + overrides
        rc = subprocess.run(cmd, cwd=cwd_train, env=env).returncode
        if rc != 0:
            if results_csv is not None and hp_cols is not None:
                append_search_results_csv(
                    results_csv,
                    cfg=cfg,
                    cfg_idx=cfg_idx,
                    seed=seed,
                    profile=profile,
                    fold_i=fold_i,
                    hp_cols=hp_cols,
                    run_dir=run_dir,
                    ckpt_path=ckpt_path,
                    metrics_path=metrics_path,
                    n_infer_episodes=n_infer_episodes,
                    status=f"train_failed_{rc}",
                )
            return None, read_val_loss_final(run_dir), rc, 0

        if not ckpt_path.is_file():
            if results_csv is not None and hp_cols is not None:
                append_search_results_csv(
                    results_csv,
                    cfg=cfg,
                    cfg_idx=cfg_idx,
                    seed=seed,
                    profile=profile,
                    fold_i=fold_i,
                    hp_cols=hp_cols,
                    run_dir=run_dir,
                    ckpt_path=ckpt_path,
                    metrics_path=metrics_path,
                    n_infer_episodes=n_infer_episodes,
                    status="no_checkpoint",
                )
            return None, read_val_loss_final(run_dir), 1, 0
    else:
        print(
            f"[random_search] infer-only cfg_idx={cfg_idx} seed={seed} profile={profile}"
        )

    print_trial_configuration(
        f"[random_search] infer cfg_idx={cfg_idx} seed={seed} profile={profile}",
        cfg,
        [
            f"checkpoint={ckpt_path}",
            f"metrics_json={metrics_path}",
            f"n_infer_episodes={n_infer_episodes}",
            f"seed={seed}",
        ],
        run_dir,
    )
    rc2 = run_infer_subprocess(
        py,
        infer_py,
        cwd_train,
        env,
        ckpt_path,
        metrics_path,
        n_infer_episodes,
        seed,
        n_train_val_episodes=n_train_val_episodes,
        train_val_eval_seed_offset=train_val_eval_seed_offset,
        skip_inference_videos=skip_inference_videos,
    )
    val_loss = read_val_loss_final(run_dir)
    if rc2 != 0 or not metrics_path.is_file():
        if results_csv is not None and hp_cols is not None:
            append_search_results_csv(
                results_csv,
                cfg=cfg,
                cfg_idx=cfg_idx,
                seed=seed,
                profile=profile,
                fold_i=fold_i,
                hp_cols=hp_cols,
                run_dir=run_dir,
                ckpt_path=ckpt_path,
                metrics_path=metrics_path,
                n_infer_episodes=n_infer_episodes,
                status=f"infer_failed_{rc2}",
            )
        return None, val_loss, rc2, target_epochs

    success_rate = read_test_success_rate(run_dir)
    if results_csv is not None and hp_cols is not None:
        append_search_results_csv(
            results_csv,
            cfg=cfg,
            cfg_idx=cfg_idx,
            seed=seed,
            profile=profile,
            fold_i=fold_i,
            hp_cols=hp_cols,
            run_dir=run_dir,
            ckpt_path=ckpt_path,
            metrics_path=metrics_path,
            n_infer_episodes=n_infer_episodes,
            status="ok",
        )
    return success_rate, val_loss, 0, target_epochs


# Alias kompatibilitas lama
def run_search_trial_training(
    *,
    cfg: Dict[str, Any],
    run_dir: pathlib.Path,
    py: str,
    train_py: pathlib.Path,
    cwd_train: str,
    seed: int,
    profile: str,
    train_eps: List[int],
    val_eps: List[int],
    zarr_rel: str,
    checkpoint_every: int,
    dataloader_num_workers: int,
    enable_early_stop: bool = True,
    early_stop_rollout_every: int = 200,
) -> Tuple[Optional[float], int, int]:
    sr, val_loss, rc, ep = run_search_trial(
        cfg=cfg,
        run_dir=run_dir,
        py=py,
        train_py=train_py,
        infer_py=pathlib.Path(""),
        cwd_train=cwd_train,
        seed=seed,
        profile=profile,
        train_eps=train_eps,
        val_eps=val_eps,
        zarr_rel=zarr_rel,
        checkpoint_every=checkpoint_every,
        dataloader_num_workers=dataloader_num_workers,
        enable_early_stop=enable_early_stop,
        early_stop_rollout_every=early_stop_rollout_every,
    )
    return val_loss, rc, ep
