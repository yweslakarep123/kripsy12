#!/usr/bin/env python3
"""
Konversi dataset Minari D4RL Kitchen (mis. ``D4RL/kitchen/complete-v2``) ke zarr
format ``KitchenDataset`` / ``ReplayBuffer``: ``state`` (T,59), ``action`` (T,9),
dan opsional ``point_cloud`` (T,512,3) kecuali ``--no-point-cloud``.

Dataset Minari hanya menyimpan vektor ``observation`` 59-d (bukan point cloud).
Skrip ini merekonstruksi ``qpos``/``qvel`` MuJoCo dari vektor tersebut (sesuai
dokumentasi ``KitchenEnv``), memanggil ``mj_forward``, lalu membangkitkan awan
titik dengan pipeline yang sama dengan ``FrankaKitchenPointCloudEnv``.

Catatan:
- Observasi D4RL mengandung noise; rekonstruksi fisika mendekati, tidak identik.
- Setelah ekspor, latih dengan: ``task.dataset.zarr_path=<path hasil>``.

Contoh:
  python scripts/export_minari_kitchen_to_flowpolicy_zarr.py \\
    --out FlowPolicy/data/kitchen_complete_from_minari.zarr \\
    --minari-id D4RL/kitchen/complete-v2 \\
    --device cuda:0 --sampling fps
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

import mujoco
import numpy as np
import zarr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "FlowPolicy"
sys.path.insert(0, str(PKG_ROOT))
os.chdir(PKG_ROOT)

import gymnasium as gym  # noqa: E402
import gymnasium_robotics  # noqa: E402
import minari  # noqa: E402

from flow_policy_3d.common.replay_buffer import ReplayBuffer  # noqa: E402
from flow_policy_3d.gym_util.mujoco_native_point_cloud import (  # noqa: E402
    NativeMuJoCoPointCloudGenerator,
)

# Sama dengan FrankaKitchenPointCloudEnv (tanpa mengimpor modul itu → hindari pytorch3d).
_KITCHEN_PC_BOUNDS = {
    "default": {
        "min_bound": [-2.0, -2.0, 0.0],
        "max_bound": [2.0, 2.0, 2.8],
    }
}


def _unwrap_kitchen_env(env: gym.Env):
    e = env
    for _ in range(12):
        if not hasattr(e, "unwrapped"):
            break
        u = e.unwrapped
        if u is e:
            break
        e = u
    return e


def obs59_to_qpos_qvel(obs_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pisahkan vektor observation (59,) menjadi qpos (30,) dan qvel (29,)."""
    o = np.asarray(obs_vec, dtype=np.float64).reshape(59)
    robot_qpos = o[:9]
    robot_qvel = o[9:18]
    obj_qpos = o[18:39]
    obj_qvel = o[39:59]
    qpos = np.concatenate([robot_qpos, obj_qpos])
    qvel = np.concatenate([robot_qvel, obj_qvel])
    return qpos, qvel


def set_sim_state(kitchen_env, obs_vec: np.ndarray) -> None:
    qpos, qvel = obs59_to_qpos_qvel(obs_vec)
    kitchen_env.data.qpos[:] = qpos
    kitchen_env.data.qvel[:] = qvel
    mujoco.mj_forward(kitchen_env.model, kitchen_env.data)


def downsample_point_cloud(
    pts: np.ndarray,
    num_points: int,
    method: str,
    device: str,
) -> np.ndarray:
    """Turunkan jumlah titik. Mode ``fps`` memakai PyTorch3D jika terpasang; jika tidak, memakai FPS iteratif murni PyTorch."""
    if num_points == "all" or pts.shape[0] <= num_points:
        d = pts.shape[-1]
        if pts.shape[0] < num_points:
            pts = np.concatenate(
                [pts, np.zeros((num_points - pts.shape[0], d), dtype=pts.dtype)], axis=0
            )
        return pts
    if method == "uniform":
        idx = np.random.choice(pts.shape[0], num_points, replace=False)
        return pts[idx]
    if method == "fps":
        from flow_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling

        return point_cloud_sampling(pts, num_points, "fps", device=device)
    raise ValueError(f"Unknown sampling method: {method}")


def build_point_cloud(
    pc_gen: NativeMuJoCoPointCloudGenerator,
    data,
    _model,
    *,
    use_point_crop: bool,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    num_points: int,
    device: str,
    sampling: str,
) -> np.ndarray:
    pts, _ = pc_gen.generate(data, use_rgb=False)
    if use_point_crop:
        mask = np.all(pts > min_bound, axis=1)
        pts = pts[mask]
        mask = np.all(pts < max_bound, axis=1)
        pts = pts[mask]
    if pts.shape[0] == 0:
        pts = np.zeros((1, 3), dtype=np.float32)
    pts = downsample_point_cloud(pts, num_points, sampling, device)
    return pts.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--minari-id",
        type=str,
        default="D4RL/kitchen/complete-v2",
        help="ID dataset Minari (harus FrankaKitchen / D4RL kitchen).",
    )
    p.add_argument("--out", type=str, required=True, help="Direktori zarr keluaran")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-points", type=int, default=512)
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Batasi jumlah episode (untuk uji cepat).",
    )
    p.add_argument(
        "--sampling",
        choices=("uniform", "fps"),
        default="uniform",
        help="fps = farthest point sampling (PyTorch3D jika ada, jika tidak fallback PyTorch); "
        "uniform = tanpa ketergantungan tambahan, distribusi berbeda dari training bila data asli memakai fps.",
    )
    p.add_argument(
        "--no-point-crop",
        action="store_true",
        help="Matikan crop bounding box (sama dengan use_point_crop=False).",
    )
    p.add_argument(
        "--no-point-cloud",
        action="store_true",
        help="Hanya ekspor state (59-d) + action; tanpa point_cloud (state-based FlowPolicy).",
    )
    args = p.parse_args()

    gym.register_envs(gymnasium_robotics)
    dataset = minari.load_dataset(args.minari_id)
    base_env = dataset.recover_environment()
    ke = _unwrap_kitchen_env(base_env)

    b = _KITCHEN_PC_BOUNDS["default"]
    min_bound = np.array(b["min_bound"], dtype=np.float32)
    max_bound = np.array(b["max_bound"], dtype=np.float32)
    use_point_crop = not args.no_point_crop

    skip_pc = bool(args.no_point_cloud)
    pc_gen = None
    if not skip_pc:
        cam_names = ["left_cap", "right_cap"]
        img_size = 128
        pc_gen = NativeMuJoCoPointCloudGenerator(
            ke.model, cam_names=cam_names, img_size=img_size
        )

    out_path = os.path.expanduser(args.out)
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store)
    buffer = ReplayBuffer.create_empty_zarr(root=root)

    n_eps = len(dataset) if args.max_episodes is None else min(
        len(dataset), args.max_episodes
    )
    for ep_idx in range(n_eps):
        ep = dataset[ep_idx]
        obs_flat = ep.observations["observation"]
        actions = ep.actions
        n_act = len(actions)
        states: list[np.ndarray] = []
        clouds: list[np.ndarray] = []
        acts: list[np.ndarray] = []
        for t in range(n_act):
            o = obs_flat[t]
            states.append(np.asarray(o, dtype=np.float32).reshape(59))
            acts.append(np.asarray(actions[t], dtype=np.float32))
            if not skip_pc:
                set_sim_state(ke, o)
                pc = build_point_cloud(
                    pc_gen,
                    ke.data,
                    ke.model,
                    use_point_crop=use_point_crop,
                    min_bound=min_bound,
                    max_bound=max_bound,
                    num_points=args.num_points,
                    device=args.device,
                    sampling=args.sampling,
                )
                clouds.append(pc)
        ep_data = {
            "state": np.stack(states, axis=0),
            "action": np.stack(acts, axis=0),
        }
        if not skip_pc:
            ep_data["point_cloud"] = np.stack(clouds, axis=0)
        buffer.add_episode(ep_data)
        print(f"episode {ep_idx + 1}/{n_eps} steps={n_act}")

    if pc_gen is not None:
        pc_gen.close()
    base_env.close()
    print(f"Saved zarr with {buffer.n_episodes} episodes -> {out_path}")


if __name__ == "__main__":
    main()
