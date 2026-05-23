#!/usr/bin/env python3
"""Jalankan satu trial training random search — override identik dengan baseline."""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from experiment_constants import CSV_HPARAM_KEYS
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
    """Latih satu konfigurasi search (satu shot, resume jika ckpt ada).

    Kembalikan ``(val_loss, returncode, target_epochs)``.
    """
    run_dir = pathlib.Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    target_epochs = int(cfg["training.num_epochs"])

    tf = run_dir / "training_final.json"
    if tf.is_file():
        v = read_val_loss_final(run_dir)
        return v, 0, target_epochs

    ckpt = run_dir / "checkpoints" / "latest.ckpt"
    resume_training = bool(ckpt.is_file())

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

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")

    print_trial_configuration(
        f"[random_search] train cfg_idx={cfg['cfg_idx']} seed={seed} profile={profile}"
        + (f" (resume={resume_training})" if resume_training else ""),
        cfg,
        overrides,
        run_dir,
    )

    cmd = [py, str(train_py)] + overrides
    rc = subprocess.run(cmd, cwd=cwd_train, env=env).returncode
    v = read_val_loss_final(run_dir)
    ep = target_epochs if rc == 0 else 0
    return v, int(rc), ep
