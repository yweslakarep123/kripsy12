"""
Inferensi Franka Kitchen dengan metrik k1–k4, latensi, dan trade_off (tanpa Hydra CLI).

Contoh:
  python infer_kitchen.py --checkpoint runs/foo/checkpoints/latest.ckpt \\
    --metrics-json runs/foo/metrics.json --n-infer-episodes 50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch

if __name__ == "__main__":
    _root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_root))
    os.chdir(str(_root))

from train import TrainFlowPolicyWorkspace  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--metrics-json", type=str, required=True)
    p.add_argument("--n-infer-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-steps", type=int, default=20)
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
    runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=out_parent,
        eval_episodes=args.n_infer_episodes,
    )
    metrics = runner.run_eval_metrics(
        policy,
        warmup_predict_steps=args.warmup_steps,
        eval_seed=args.seed,
        log_video=False,
    )

    serializable = {k: v for k, v in metrics.items() if k != "sim_video_eval"}
    path = pathlib.Path(args.metrics_json).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    main()
