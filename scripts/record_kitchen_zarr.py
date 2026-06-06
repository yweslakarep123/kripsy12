#!/usr/bin/env python3
"""
Kumpulkan rollout Franka Kitchen (policy acak) ke zarr untuk KitchenDataset.
Ganti sampling policy dengan demonstrasi ahli / BC jika tersedia.

Untuk dataset Minari D4RL (mis. ``D4RL/kitchen/complete-v2``) tanpa point cloud
tersimpan, gunakan skrip ``scripts/export_minari_kitchen_to_flowpolicy_zarr.py``
agar trajektori diubah ke format zarr FlowPolicy (state, action, point cloud).

Contoh:
  cd FlowPolicy && python scripts/record_kitchen_zarr.py \\
    --out FlowPolicy/data/franka_kitchen_sequential4_expert.zarr \\
    --episodes 50 --sequential --seed 0
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

import numpy as np
import zarr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "FlowPolicy"
sys.path.insert(0, str(PKG_ROOT))
os.chdir(PKG_ROOT)

from flow_policy_3d.common.replay_buffer import ReplayBuffer  # noqa: E402
from flow_policy_3d.env.franka_kitchen import FrankaKitchenPointCloudEnv  # noqa: E402


SEQUENTIAL_FOUR = [
    "microwave",
    "kettle",
    "light switch",
    "slide cabinet",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Direktori zarr keluaran")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Subtask kitchen (mode non-urut). Kosong = semua task.",
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help=f"Empat task berurutan: {' → '.join(SEQUENTIAL_FOUR)}",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-steps", type=int, default=280)
    p.add_argument("--num-points", type=int, default=512)
    p.add_argument(
        "--point-cloud",
        action="store_true",
        help="Sertakan point_cloud di zarr (default: state-only 59-d).",
    )
    args = p.parse_args()

    if args.sequential and args.tasks:
        p.error("Gunakan --sequential saja, atau --tasks, jangan keduanya.")

    out_path = os.path.expanduser(args.out)
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    rng = np.random.default_rng(args.seed)
    obs_mode = "point_cloud" if args.point_cloud else "state"
    env_kw = dict(
        device=args.device,
        obs_mode=obs_mode,
        num_points=args.num_points,
        max_episode_steps=args.max_steps,
    )
    if args.sequential:
        env = FrankaKitchenPointCloudEnv(
            task_completion_order=SEQUENTIAL_FOUR,
            **env_kw,
        )
    else:
        tasks = args.tasks if args.tasks else None
        env = FrankaKitchenPointCloudEnv(
            tasks_to_complete=tasks,
            **env_kw,
        )

    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store)
    buffer = ReplayBuffer.create_empty_zarr(root=root)

    try:
        for ep in range(args.episodes):
            ret = env.reset(seed=args.seed + ep)
            obs = ret[0] if isinstance(ret, tuple) else ret
            done = False
            states, actions, clouds = [], [], []
            while not done:
                states.append(obs["agent_pos"].copy())
                if args.point_cloud:
                    clouds.append(obs["point_cloud"].copy())
                a = env.action_space.sample()
                a = rng.normal(0, 0.3, size=a.shape).astype(np.float32)
                a = np.clip(a, env.action_space.low, env.action_space.high)
                actions.append(a.copy())
                obs, _r, done, _info = env.step(a)
            if len(actions) == 0:
                continue
            ep_data = {
                "state": np.stack(states, axis=0).astype(np.float32),
                "action": np.stack(actions, axis=0).astype(np.float32),
            }
            if args.point_cloud:
                ep_data["point_cloud"] = np.stack(clouds, axis=0).astype(np.float32)
            buffer.add_episode(ep_data)
            print(f"episode {ep+1}/{args.episodes} length={len(actions)}")
    finally:
        env.close()

    print(f"Saved zarr with {buffer.n_episodes} episodes -> {out_path}")


if __name__ == "__main__":
    main()
