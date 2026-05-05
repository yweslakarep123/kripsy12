"""
Pembentukan lipatan validasi silang pada level episode.

Contoh: 19 episode, 1 test terholdout, 5 fold pada 18 episode sisa
→ setiap fold ~15 train, ~3–4 val (bergantung pemisahan).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np


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
