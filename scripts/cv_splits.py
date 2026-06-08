"""
Partisi train/val/test pada level episode demo Kitchen (.mjl).

Split aktif untuk eksperimen: holdout ``n_test_holdout`` episode (default 50,
selaras jumlah episode inferensi simulasi), sisanya ``train_frac`` / val.
Indeks episode = urutan sorted ``*/*.mjl`` di folder dataset (deterministik).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def count_kitchen_mjl_episodes(dataset_dir: Path) -> int:
    """Hitung jumlah file ``*/*.mjl`` (satu file = satu episode)."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir tidak ada: {dataset_dir}")
    n = len(sorted(dataset_dir.glob("*/*.mjl")))
    if n == 0:
        raise FileNotFoundError(
            f"Tidak ada file */*.mjl di {dataset_dir.resolve()}"
        )
    return n


def build_kitchen_demo_split(
    n_episodes: int,
    *,
    n_test_holdout: int = 50,
    train_frac: float = 0.8,
    seed: int = 12345,
) -> Dict[str, Any]:
    """
    Satu partisi episode-level untuk Kitchen MJL.

    1. Acak ``n_episodes`` indeks dengan ``seed``.
    2. ``n_test_holdout`` pertama → test (tidak dipakai training/val).
    3. Sisanya → ``train_frac`` train, remainder val.

    Contoh 605 episode, holdout 50, train_frac 0.8:
      test=50, train=444, val=111.
    """
    if n_episodes < 3:
        raise ValueError(f"n_episodes minimal 3, dapat {n_episodes}")
    if n_test_holdout < 0 or n_test_holdout >= n_episodes:
        raise ValueError(
            f"n_test_holdout ({n_test_holdout}) harus dalam [0, {n_episodes - 1})"
        )
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac harus di (0, 1), dapat {train_frac}")

    n_train_val = n_episodes - n_test_holdout
    if n_train_val < 2:
        raise ValueError(
            f"Perlu minimal 2 episode train+val, dapat {n_train_val}"
        )

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n_episodes).tolist()
    test_episodes = sorted(perm[:n_test_holdout])
    rest = perm[n_test_holdout:]
    n_train = int(round(n_train_val * train_frac))
    n_train = max(1, min(n_train, n_train_val - 1))
    n_val = n_train_val - n_train
    train_episodes = sorted(rest[:n_train])
    val_episodes = sorted(rest[n_train:])

    return {
        "fold": 0,
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "test_episodes": test_episodes,
        "n_episodes": int(n_episodes),
        "n_test_holdout": int(n_test_holdout),
        "train_frac": float(train_frac),
        "split_seed": int(seed),
        "n_train": len(train_episodes),
        "n_val": len(val_episodes),
        "n_test": len(test_episodes),
    }


def build_single_train_val_split(
    n_episodes: int = 19,
    held_out_test: int = 1,
    *,
    n_grid_partitions: int = 5,
    partition_index: int = 0,
    seed: int = 12345,
) -> Dict[str, Any]:
    """
    Legacy k-fold geometry (19 episode). Prefer ``build_kitchen_demo_split``.
    """
    folds = build_cv_splits(
        n_episodes=n_episodes,
        n_folds=n_grid_partitions,
        held_out_test=held_out_test,
        seed=seed,
    )
    if partition_index < 0 or partition_index >= len(folds):
        raise ValueError(
            f"partition_index {partition_index} tidak valid "
            f"(ada {len(folds)} partisi)."
        )
    return folds[partition_index]


def build_cv_splits(
    n_episodes: int = 19,
    n_folds: int = 5,
    held_out_test: int = 1,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Legacy k-fold splits."""
    if held_out_test < 1:
        raise ValueError("held_out_test minimal 1")
    if n_episodes < held_out_test + n_folds:
        raise ValueError(
            f"n_episodes ({n_episodes}) terlalu kecil untuk test={held_out_test} "
            f"dan {n_folds} fold."
        )

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(np.arange(n_episodes)).tolist()
    test_episodes = sorted(perm[:held_out_test])
    rest = np.array(perm[held_out_test:], dtype=int)

    splits = np.array_split(rest, n_folds)
    folds: List[Dict[str, Any]] = []
    for k in range(n_folds):
        val_arr = splits[k]
        train_arr = np.concatenate([splits[i] for i in range(n_folds) if i != k])
        folds.append(
            {
                "fold": k,
                "train_episodes": sorted(train_arr.astype(int).tolist()),
                "val_episodes": sorted(val_arr.astype(int).tolist()),
                "test_episodes": list(test_episodes),
            }
        )
    return folds


def save_episode_split(path: str, split: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
    payload = {"meta": meta or {}, "split": split}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_splits(path: str, folds: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    payload = {"meta": meta, "folds": folds}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
