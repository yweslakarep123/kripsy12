from __future__ import annotations

import json
import pathlib
import wandb
import time
import numpy as np
import torch
import tqdm

from flow_policy_3d.env.franka_kitchen import FrankaKitchenPointCloudEnv
from flow_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from flow_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from flow_policy_3d.policy.base_policy import BasePolicy
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.env_runner.base_runner import BaseRunner
import flow_policy_3d.common.logger_util as logger_util
from termcolor import cprint


class KitchenRunner(BaseRunner):
    """Urutan sub-tugas untuk metrik success_rate_k1…k4 (Kitchen-Complete)."""

    K_LEVEL_SPECS = (
        frozenset({"microwave"}),
        frozenset({"microwave", "light switch"}),
        frozenset({"microwave", "light switch", "kettle"}),
        frozenset({"microwave", "light switch", "kettle", "slide cabinet"}),
    )

    def __init__(
        self,
        output_dir,
        eval_episodes=20,
        max_steps=280,
        n_obs_steps=8,
        n_action_steps=8,
        fps=12,
        crf=22,
        render_size=84,
        tqdm_interval_sec=5.0,
        n_envs=None,
        task_name=None,
        n_train=None,
        n_test=None,
        device="cuda",
        use_point_crop=True,
        num_points=512,
        tasks_to_complete=None,
        task_completion_order=None,
        terminate_on_tasks_completed=True,
    ):
        super().__init__(output_dir)
        self.task_name = task_name

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    FrankaKitchenPointCloudEnv(
                        tasks_to_complete=tasks_to_complete,
                        task_completion_order=task_completion_order,
                        device=device,
                        use_point_crop=use_point_crop,
                        num_points=num_points,
                        terminate_on_tasks_completed=terminate_on_tasks_completed,
                        max_episode_steps=max_steps,
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method="sum",
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self._env_closed = False

    def close(self):
        """Tutup sim + renderer MuJoCo (EGL) agar tidak bergantung pada __del__ saat shutdown."""
        if self._env_closed:
            return
        self._env_closed = True
        # #region agent log
        try:
            p = {
                "sessionId": "675d16",
                "location": "kitchen_runner.py:close",
                "message": "KitchenRunner.close calling env.close()",
                "data": {"runId": "post-fix"},
                "timestamp": int(time.time() * 1000),
                "hypothesisId": "H1",
            }
            with open(
                "/home/daffa/Documents/kripsy12/.cursor/debug-675d16.log", "a"
            ) as _df:
                _df.write(json.dumps(p) + "\n")
        except Exception:
            pass
        # #endregion
        self.env.close()

    @staticmethod
    def _completion_set_from_info(info: dict) -> set:
        """Gabungkan semua task yang tercatat di riwayat obs langkah terakhir."""
        raw = info.get("episode_task_completions")
        if raw is None:
            return set()
        if isinstance(raw, np.ndarray):
            seq = raw.ravel().tolist()
        elif isinstance(raw, (list, tuple)):
            seq = list(raw)
        else:
            seq = [raw]
        merged = set()
        for item in seq:
            if item is None:
                continue
            if isinstance(item, str):
                merged.add(item)
            elif isinstance(item, (list, tuple, set)):
                merged.update(str(x) for x in item)
            elif isinstance(item, np.ndarray):
                merged.update(str(x) for x in item.tolist())
            else:
                merged.add(str(item))
        return merged

    @staticmethod
    def _save_rgb_video_mp4(path: pathlib.Path, tc_hwc: np.ndarray, fps: float) -> None:
        """Simpan tensor (T, C, H, W) uint8 ke MP4."""
        if tc_hwc.ndim != 4:
            raise ValueError(f"video shape tidak didukung: {tc_hwc.shape}")
        thwc = np.transpose(tc_hwc, (0, 2, 3, 1))
        if thwc.dtype != np.uint8:
            thwc = np.clip(thwc, 0, 255).astype(np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        imageio.mimsave(str(path), thwc, fps=float(fps))

    def run_eval_metrics(
        self,
        policy: BasePolicy,
        *,
        warmup_predict_steps: int = 20,
        eval_seed: int = 0,
        log_video: bool = False,
        n_episodes: int | None = None,
        save_inference_videos_dir: str | pathlib.Path | None = None,
    ):
        """
        Evaluasi dengan warmup GPU, latensi per langkah predict_action, dan success k1–k4.
        Tidak mengandalkan W&B kecuali log_video=True (video terakhir).

        Args:
            n_episodes: Override jumlah episode evaluasi (default: ``self.eval_episodes``).
            save_inference_videos_dir: Jika diisi, simpan satu MP4 per episod (inferensi).
        """
        device = policy.device
        env = self.env
        latencies_ms: list[float] = []
        current_ep_lat_ms: list[float] = []

        def predict_timed(obs_dict_input):
            with torch.no_grad():
                t0 = time.perf_counter()
                action_dict = policy.predict_action(obs_dict_input)
                t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000.0
            latencies_ms.append(dt_ms)
            current_ep_lat_ms.append(dt_ms)
            return action_dict

        obs = env.reset(seed=eval_seed)
        policy.reset()
        np_obs_dict = dict(obs)
        obs_dict = dict_apply(
            np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
        )
        obs_dict_input = {
            "point_cloud": obs_dict["point_cloud"].unsqueeze(0),
            "agent_pos": obs_dict["agent_pos"].unsqueeze(0),
        }
        for _ in range(max(0, warmup_predict_steps)):
            predict_timed(obs_dict_input)

        latencies_ms.clear()
        current_ep_lat_ms.clear()

        ep_success_levels = []
        per_episode_mean_inference_latency_ms: list[float] = []
        n_eps = int(self.eval_episodes if n_episodes is None else n_episodes)
        video_root = (
            pathlib.Path(save_inference_videos_dir).resolve()
            if save_inference_videos_dir
            else None
        )

        for episode_idx in tqdm.tqdm(
            range(n_eps),
            desc=f"EvalMetrics FrankaKitchen {self.task_name}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            current_ep_lat_ms.clear()
            obs = env.reset(seed=int(eval_seed + episode_idx))
            policy.reset()

            done = False
            last_completions = set()
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )
                obs_dict_input = {
                    "point_cloud": obs_dict["point_cloud"].unsqueeze(0),
                    "agent_pos": obs_dict["agent_pos"].unsqueeze(0),
                }
                action_dict = predict_timed(obs_dict_input)
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )
                action = np_action_dict["action"].squeeze(0)

                obs, reward, done, info = env.step(action)
                done = np.all(done)
                last_completions |= self._completion_set_from_info(info)

            levels_met = [
                spec.issubset(last_completions) for spec in self.K_LEVEL_SPECS
            ]
            ep_success_levels.append(levels_met)

            ep_mean = (
                float(np.mean(current_ep_lat_ms)) if current_ep_lat_ms else 0.0
            )
            per_episode_mean_inference_latency_ms.append(ep_mean)

            if video_root is not None:
                videos = env.env.get_video()
                if len(videos.shape) == 5:
                    videos = videos[:, 0]
                out_mp4 = video_root / f"infer_ep_{episode_idx:03d}.mp4"
                self._save_rgb_video_mp4(out_mp4, videos, self.fps)

        sr = np.asarray(ep_success_levels, dtype=np.float64)
        mean_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        std_lat = float(np.std(latencies_ms)) if latencies_ms else 0.0
        ep_means_arr = np.asarray(per_episode_mean_inference_latency_ms, dtype=np.float64)
        mean_episode_mean_lat = float(np.mean(ep_means_arr)) if len(ep_means_arr) else 0.0
        std_episode_mean_lat = float(np.std(ep_means_arr)) if len(ep_means_arr) else 0.0
        # Sukses penuh Kitchen-Complete (empat sub-tugas terurut) = level k4.
        success_total_pct = float(sr[:, 3].mean() * 100.0)
        out = {
            "success_rate_total": success_total_pct,
            "success_rate_k1": float(sr[:, 0].mean() * 100.0),
            "success_rate_k2": float(sr[:, 1].mean() * 100.0),
            "success_rate_k3": float(sr[:, 2].mean() * 100.0),
            "success_rate_k4": float(sr[:, 3].mean() * 100.0),
            "mean_inference_latency_ms": mean_lat,
            "std_inference_latency_ms": std_lat,
            "per_episode_mean_inference_latency_ms": per_episode_mean_inference_latency_ms,
            "mean_episode_mean_inference_latency_ms": mean_episode_mean_lat,
            "std_episode_mean_inference_latency_ms": std_episode_mean_lat,
            "n_infer_episodes": int(n_eps),
        }
        out["trade_off"] = (
            float(out["success_rate_k4"] / mean_lat) if mean_lat > 1e-9 else 0.0
        )
        out["trade_off_episode_latency"] = (
            float(out["success_rate_k4"] / mean_episode_mean_lat)
            if mean_episode_mean_lat > 1e-9
            else 0.0
        )

        if log_video:
            videos = env.env.get_video()
            if len(videos.shape) == 5:
                videos = videos[:, 0]
            out["sim_video_eval"] = wandb.Video(videos, fps=self.fps, format="mp4")

        cprint(
            f"[run_eval_metrics] k4={out['success_rate_k4']:.2f}% "
            f"lat_ms={mean_lat:.3f}±{std_lat:.3f}",
            "green",
        )
        # #region agent log
        try:
            p = {
                "sessionId": "675d16",
                "location": "kitchen_runner.py:run_eval_metrics_return",
                "message": "run_eval_metrics returning (no env.close in this method)",
                "data": {},
                "timestamp": int(time.time() * 1000),
                "hypothesisId": "H1",
            }
            with open(
                "/home/daffa/Documents/kripsy12/.cursor/debug-675d16.log", "a"
            ) as _df:
                _df.write(json.dumps(p) + "\n")
        except Exception:
            pass
        # #endregion
        return out

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        all_time = []
        env = self.env

        for episode_idx in tqdm.tqdm(
            range(self.eval_episodes),
            desc=f"Eval FrankaKitchen {self.task_name}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            actual_step_count = 0
            total_time = 0
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input["point_cloud"] = obs_dict["point_cloud"].unsqueeze(0)
                    obs_dict_input["agent_pos"] = obs_dict["agent_pos"].unsqueeze(0)
                    start_time = time.time()
                    action_dict = policy.predict_action(obs_dict_input)
                    end_time = time.time()
                    total_time += end_time - start_time

                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )
                action = np_action_dict["action"].squeeze(0)

                obs, reward, done, info = env.step(action)

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info["success"])
                actual_step_count += 1

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            all_time.append(total_time / max(actual_step_count, 1))

        log_data = dict()

        log_data["mean_traj_rewards"] = np.mean(all_traj_rewards)
        log_data["mean_success_rates"] = np.mean(all_success_rates)
        log_data["mean_time"] = np.mean(all_time)

        log_data["test_mean_score"] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", "green")

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data["SR_test_L3"] = self.logger_util_test.average_of_largest_K()
        log_data["SR_test_L5"] = self.logger_util_test10.average_of_largest_K()

        # Video rollout training tidak di-log ke media; gunakan infer_kitchen + MP4 inferensi.
        _ = env.reset()

        return log_data
