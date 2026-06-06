#!/usr/bin/env python3
"""Utilitas inferensi rollout — dipakai baseline, random search, dan rerun."""

from __future__ import annotations

import pathlib
import subprocess


def run_infer_subprocess(
    py: str,
    infer_py: pathlib.Path,
    cwd_train: str,
    env: dict,
    ckpt_path: pathlib.Path,
    metrics_path: pathlib.Path,
    n_infer_episodes: int,
    seed: int,
    *,
    n_train_val_episodes: int,
    train_val_eval_seed_offset: int,
    skip_inference_videos: bool = False,
) -> int:
    cmd = [
        py,
        str(infer_py),
        "--checkpoint",
        str(ckpt_path),
        "--metrics-json",
        str(metrics_path),
        "--n-train-val-episodes",
        str(int(n_train_val_episodes)),
        "--train-val-eval-seed-offset",
        str(int(train_val_eval_seed_offset)),
        "--n-infer-episodes",
        str(n_infer_episodes),
        "--seed",
        str(seed),
        "--warmup-steps",
        "20",
    ]
    if skip_inference_videos:
        cmd.append("--skip-inference-videos")
    else:
        vdir = pathlib.Path(metrics_path).parent / "inference_videos"
        cmd.extend(["--inference-videos-dir", str(vdir.resolve())])
    return subprocess.run(cmd, cwd=cwd_train, env=env).returncode
