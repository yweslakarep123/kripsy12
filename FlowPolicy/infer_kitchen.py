"""
Inferensi Franka Kitchen dengan metrik test k1–k4 (per-task), success total,
latensi (global + per-episod), waktu pengerjaan, trade_off, dan MP4 inferensi.

Evaluasi simulasi hanya pada fase test (default 50 episode). Training tidak
menjalankan eval success rate / latensi.

Contoh:
  python infer_kitchen.py --checkpoint runs/foo/checkpoints/latest.ckpt \\
    --metrics-json runs/foo/metrics.json --n-infer-episodes 50 --seed 42
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch

if __name__ == "__main__":
    _root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_root))
    os.chdir(str(_root))

from train import TrainFlowPolicyWorkspace  # noqa: E402


def _prefix_metrics(prefix: str, m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if k == "sim_video_eval":
            continue
        out[f"{prefix}_{k}"] = v
    return out


def _legacy_from_test(test: dict) -> dict:
    """Kunci tanpa prefix (kompatibel parser lama) = fase test / inferensi utama."""
    out = {
        "success_rate_total": test.get("success_rate_total"),
        "std_success_rate_total": test.get("std_success_rate_total"),
        "success_rate_k1": test.get("success_rate_k1"),
        "success_rate_k2": test.get("success_rate_k2"),
        "success_rate_k3": test.get("success_rate_k3"),
        "success_rate_k4": test.get("success_rate_k4"),
        "std_success_rate_k1": test.get("std_success_rate_k1"),
        "std_success_rate_k2": test.get("std_success_rate_k2"),
        "std_success_rate_k3": test.get("std_success_rate_k3"),
        "std_success_rate_k4": test.get("std_success_rate_k4"),
        "mean_inference_latency_ms": test.get("mean_inference_latency_ms"),
        "std_inference_latency_ms": test.get("std_inference_latency_ms"),
        "mean_episode_mean_inference_latency_ms": test.get(
            "mean_episode_mean_inference_latency_ms"
        ),
        "std_episode_mean_inference_latency_ms": test.get(
            "std_episode_mean_inference_latency_ms"
        ),
        "mean_execution_time_ms": test.get("mean_execution_time_ms"),
        "std_execution_time_ms": test.get("std_execution_time_ms"),
        "total_execution_time_ms": test.get("total_execution_time_ms"),
        "mean_all_tasks_execution_time_ms": test.get(
            "mean_all_tasks_execution_time_ms"
        ),
        "n_infer_episodes": test.get("n_infer_episodes"),
        "trade_off": test.get("trade_off"),
        "trade_off_episode_latency": test.get("trade_off_episode_latency"),
        "per_episode_mean_inference_latency_ms": test.get(
            "per_episode_mean_inference_latency_ms"
        ),
    }
    for ki in range(1, 5):
        out[f"mean_task_execution_time_ms_k{ki}"] = test.get(
            f"mean_task_execution_time_ms_k{ki}"
        )
        out[f"std_task_execution_time_ms_k{ki}"] = test.get(
            f"std_task_execution_time_ms_k{ki}"
        )
    return out


def _write_video_manifest(video_dir: pathlib.Path, n_expected: int) -> None:
    if not video_dir.is_dir():
        return
    mp4s = sorted(video_dir.glob("infer_ep_*.mp4"))
    manifest = {
        "kind": "inference_rollout",
        "n_expected_episodes": int(n_expected),
        "n_saved": len(mp4s),
        "files": [x.name for x in mp4s],
    }
    with open(video_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    # #region agent log
    def _dbg_atexit():
        try:
            p = {
                "sessionId": "675d16",
                "location": "infer_kitchen.py:atexit",
                "message": "process atexit (before full interpreter teardown)",
                "data": {},
                "timestamp": int(time.time() * 1000),
                "hypothesisId": "H2",
            }
            with open(
                "/home/daffa/Documents/kripsy12/.cursor/debug-675d16.log", "a"
            ) as f:
                f.write(json.dumps(p) + "\n")
        except Exception:
            pass

    atexit.register(_dbg_atexit)
    # #endregion
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--metrics-json", type=str, required=True)
    p.add_argument(
        "--n-train-val-episodes",
        type=int,
        default=0,
        help="Episode eval fase train/val (sim); 0 = lewati fase ini (default).",
    )
    p.add_argument("--n-infer-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument(
        "--train-val-eval-seed-offset",
        type=int,
        default=31,
        help="Offset seed eval train/val vs test agar episode tidak identik.",
    )
    p.add_argument(
        "--inference-videos-dir",
        type=str,
        default=None,
        help="Folder untuk MP4 per-episod inferensi (default: <run>/inference_videos).",
    )
    p.add_argument(
        "--skip-inference-videos",
        action="store_true",
        help="Jangan menulis MP4 inferensi (hanya metrik).",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ckpt = pathlib.Path(args.checkpoint).resolve()
    workspace = TrainFlowPolicyWorkspace.create_from_checkpoint(str(ckpt))
    cfg = workspace.cfg
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    device = torch.device(cfg.training.device)
    policy.to(device)

    import hydra

    out_parent = str(ckpt.parent.parent)
    n_max = max(int(args.n_train_val_episodes), int(args.n_infer_episodes))
    runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=out_parent,
        eval_episodes=n_max,
    )
    path = pathlib.Path(args.metrics_json).resolve()
    run_dir = path.parent
    if args.skip_inference_videos:
        video_dir_arg = None
    elif args.inference_videos_dir:
        video_dir_arg = str(pathlib.Path(args.inference_videos_dir).resolve())
    else:
        video_dir_arg = str((run_dir / "inference_videos").resolve())

    try:
        m_te = runner.run_eval_metrics(
            policy,
            warmup_predict_steps=args.warmup_steps,
            eval_seed=int(args.seed),
            log_video=False,
            n_episodes=int(args.n_infer_episodes),
            save_inference_videos_dir=video_dir_arg,
        )

        serializable = {
            **_prefix_metrics("test", m_te),
            **_legacy_from_test(m_te),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

        if video_dir_arg:
            _write_video_manifest(
                pathlib.Path(video_dir_arg), int(args.n_infer_episodes)
            )

        # #region agent log
        try:
            p = {
                "sessionId": "675d16",
                "location": "infer_kitchen.py:main_end",
                "message": "infer_kitchen main finished writing metrics",
                "data": {"runId": "post-fix"},
                "timestamp": int(time.time() * 1000),
                "hypothesisId": "H1",
            }
            with open(
                "/home/daffa/Documents/kripsy12/.cursor/debug-675d16.log", "a"
            ) as f:
                f.write(json.dumps(p) + "\n")
        except Exception:
            pass
        # #endregion
    finally:
        try:
            runner.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
