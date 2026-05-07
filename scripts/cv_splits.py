"""
Partisi train/val/test pada level episode.

Untuk eksperimen utama dipakai **satu** partisi train/val (tanpa melatih semua
lipatan k-fold). Fungsi `build_cv_splits` tetap ada untuk mereproduksi geometri
lipatan; `build_single_train_val_split` hanya mengambil satu lipatan tersebut.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np


def build_single_train_val_split(
    n_episodes: int = 19,
    held_out_test: int = 1,
    *,
    n_grid_partitions: int = 5,
    partition_index: int = 0,
    seed: int = 12345,
) -> Dict[str, Any]:
    """
    Satu partisi train/val/test — sama dengan satu elemen dari `build_cv_splits`
    dengan `n_folds=n_grid_partitions`, dipilih `partition_index`.

    Ini **bukan** pelatihan k-fold: hanya satu pemisahan episode yang dipakai.
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
    """
    Mengeluarkan daftar fold; episode test diholdout permanen sebelum CV.

    Return tiap elemen:
      fold, train_episodes, val_episodes, test_episodes
    """
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


def save_splits(path: str, folds: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    payload = {"meta": meta, "folds": folds}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
