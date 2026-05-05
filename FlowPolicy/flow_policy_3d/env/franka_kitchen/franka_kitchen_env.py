import json
import time

import gym
import gymnasium
import gymnasium_robotics
import numpy as np
from gym import spaces
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import (
    BONUS_THRESH,
    OBS_ELEMENT_GOALS,
)
from termcolor import cprint

from flow_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling
from flow_policy_3d.gym_util.mujoco_native_point_cloud import NativeMuJoCoPointCloudGenerator

# Rough workspace crop in world coordinates (tune for your camera setup)
KITCHEN_PC_BOUNDS = {
    "default": {
        "min_bound": [-2.0, -2.0, 0.0],
        "max_bound": [2.0, 2.0, 2.8],
    }
}


class FrankaKitchenPointCloudEnv(gym.Env):
    """
    FrankaKitchen (Gymnasium) dengan observasi FlowPolicy (point cloud + low-dim).

    Jika ``task_completion_order`` diisi, reward sparse hanya ketika task berikutnya
    dalam urutan itu pertama kali memenuhi threshold (task lain tidak dihitung lebih dulu).
    Episod terminasi setelah semua task dalam urutan selesai.
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 12}

    def __init__(
        self,
        tasks_to_complete=None,
        task_completion_order=None,
        device="cuda",
        use_point_crop=True,
        num_points=512,
        image_size=128,
        cam_names=None,
        terminate_on_tasks_completed=True,
        remove_task_when_completed=True,
        max_episode_steps=280,
    ):
        super().__init__()
        gymnasium.register_envs(gymnasium_robotics)

        # Urutan sequential: reward + selesai episodik hanya jika task i selesai setelah i-1.
        self._task_completion_order = (
            list(task_completion_order) if task_completion_order else None
        )
        if self._task_completion_order:
            for t in self._task_completion_order:
                if t not in OBS_ELEMENT_GOALS:
                    raise ValueError(
                        f"Unknown kitchen task {t!r}. Valid: {list(OBS_ELEMENT_GOALS.keys())}"
                    )
            tasks_for_env = list(dict.fromkeys(self._task_completion_order))
            terminate_on_tasks_completed = False
            remove_task_when_completed = False
        else:
            if tasks_to_complete is None:
                tasks_for_env = list(OBS_ELEMENT_GOALS.keys())
            else:
                tasks_for_env = list(tasks_to_complete)

        if cam_names is None:
            cam_names = ["left_cap", "right_cap"]

        self._gymnasium_env = gymnasium.make(
            "FrankaKitchen-v1",
            render_mode="rgb_array",
            tasks_to_complete=tasks_for_env,
            terminate_on_tasks_completed=terminate_on_tasks_completed,
            remove_task_when_completed=remove_task_when_completed,
            max_episode_steps=max_episode_steps,
        )
        self._ke = self._gymnasium_env.unwrapped
        self.model = self._ke.model
        self.data = self._ke.data

        self.use_point_crop = use_point_crop
        self.num_points = num_points
        self.image_size = image_size
        self._device = device

        self._pc_gen = NativeMuJoCoPointCloudGenerator(
            self.model, cam_names=cam_names, img_size=image_size
        )

        b = KITCHEN_PC_BOUNDS["default"]
        self.min_bound = np.array(b["min_bound"], dtype=np.float32)
        self.max_bound = np.array(b["max_bound"], dtype=np.float32)

        self.action_space = spaces.Box(
            low=self._gymnasium_env.action_space.low,
            high=self._gymnasium_env.action_space.high,
            dtype=np.float32,
        )
        self._n_goal_tasks = (
            len(self._task_completion_order)
            if self._task_completion_order
            else len(self._ke.goal.keys())
        )

        self._reset_sequential_state()

        obs_dim = 9
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                ),
                "point_cloud": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(num_points, 3),
                    dtype=np.float32,
                ),
            }
        )
        cprint(
            f"[FrankaKitchenPointCloudEnv] tasks={tasks_for_env} "
            f"sequential_order={self._task_completion_order} num_points={num_points}",
            "cyan",
        )

    def _reset_sequential_state(self):
        if not self._task_completion_order:
            self._sequential_idx = 0
            self._prev_dist_current_stage = np.inf
            return
        self._sequential_idx = 0
        self._prev_dist_current_stage = np.inf

    def close(self):
        self._pc_gen.close()
        self._gymnasium_env.close()
        super().close()

    def _agent_pos(self, obs_dict) -> np.ndarray:
        return obs_dict["observation"][:9].astype(np.float32)

    def _get_point_cloud(self) -> np.ndarray:
        pts, _ = self._pc_gen.generate(self.data, use_rgb=False)
        if self.use_point_crop:
            mask = np.all(pts > self.min_bound, axis=1)
            pts = pts[mask]
            mask = np.all(pts < self.max_bound, axis=1)
            pts = pts[mask]
        if pts.shape[0] == 0:
            pts = np.zeros((1, 3), dtype=np.float32)
        pts = point_cloud_sampling(
            pts, self.num_points, "fps", device=self._device
        )
        return pts.astype(np.float32)

    def _wrap_obs(self, gym_obs) -> dict:
        return {
            "agent_pos": self._agent_pos(gym_obs),
            "point_cloud": self._get_point_cloud(),
        }

    def _augment_info(self, info: dict) -> dict:
        if self._task_completion_order:
            ep = self._task_completion_order[: self._sequential_idx]
            success = float(self._sequential_idx >= len(self._task_completion_order))
        else:
            ep = info.get("episode_task_completions", [])
            success = float(len(ep) == self._n_goal_tasks)
        out = dict(info)
        out["success"] = np.array([success], dtype=np.float32)
        out["episode_task_completions"] = ep
        out["sequential_idx"] = self._sequential_idx
        # #region agent log
        try:
            self._dbg_aug_i = getattr(self, "_dbg_aug_i", 0) + 1
            if self._dbg_aug_i <= 8:
                ep_len = len(ep) if hasattr(ep, "__len__") else -1
                p = {
                    "sessionId": "74ea2d",
                    "location": "franka_kitchen_env.py:_augment_info",
                    "message": "augment_info snapshot",
                    "data": {
                        "call": self._dbg_aug_i,
                        "ep_len": ep_len,
                        "ordered": bool(self._task_completion_order),
                        "seq_idx": int(self._sequential_idx),
                    },
                    "timestamp": int(time.time() * 1000),
                    "hypothesisId": "H1",
                }
                with open(
                    "/home/daffa/Documents/FlowPolicy/.cursor/debug-74ea2d.log",
                    "a",
                ) as f:
                    f.write(json.dumps(p) + "\n")
        except Exception:
            pass
        # #endregion
        return out

    def reset(self, **kwargs):
        self._dbg_aug_i = 0
        seed = kwargs.get("seed")
        options = kwargs.get("options")
        if seed is not None or options is not None:
            obs, info = self._gymnasium_env.reset(seed=seed, options=options)
        else:
            obs, info = self._gymnasium_env.reset()
        self._reset_sequential_state()
        wrapped = self._wrap_obs(obs)
        # gym>=0.26: reset returns (obs, info)
        return wrapped, self._augment_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._gymnasium_env.step(
            np.asarray(action, dtype=np.float32)
        )

        if self._task_completion_order:
            reward = 0.0
            n = len(self._task_completion_order)
            if self._sequential_idx < n:
                task = self._task_completion_order[self._sequential_idx]
                ach = np.asarray(obs["achieved_goal"][task], dtype=np.float64)
                des = np.asarray(self._ke.goal[task], dtype=np.float64)
                dist = np.linalg.norm(ach - des)
                crossed = dist < BONUS_THRESH and self._prev_dist_current_stage >= BONUS_THRESH
                if crossed:
                    reward = 1.0
                    self._sequential_idx += 1
                    self._prev_dist_current_stage = np.inf
                else:
                    self._prev_dist_current_stage = dist
            if self._sequential_idx >= n:
                terminated = True

        done = terminated or truncated
        return self._wrap_obs(obs), reward, done, self._augment_info(info)

    def render(self, mode="rgb_array"):
        frame = self._gymnasium_env.render()
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def seed(self, seed=None):
        pass
