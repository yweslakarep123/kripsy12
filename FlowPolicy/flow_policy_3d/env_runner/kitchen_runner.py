from __future__ import annotations

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

    def run_eval_metrics(
        self,
        policy: BasePolicy,
        *,
        warmup_predict_steps: int = 20,
        eval_seed: int = 0,
        log_video: bool = False,
        n_episodes: int | None = None,
    ):
        """
        Evaluasi dengan warmup GPU, latensi per langkah predict_action, dan success k1–k4.
        Tidak mengandalkan W&B kecuali log_video=True (video terakhir).

        Args:
            n_episodes: Override jumlah episode evaluasi (default: ``self.eval_episodes``).
        """
        device = policy.device
        env = self.env
        latencies_ms = []

        def predict_timed(obs_dict_input):
            with torch.no_grad():
                t0 = time.perf_counter()
                action_dict = policy.predict_action(obs_dict_input)
                t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
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

        ep_success_levels = []
        n_eps = int(self.eval_episodes if n_episodes is None else n_episodes)

        for episode_idx in tqdm.tqdm(
            range(n_eps),
            desc=f"EvalMetrics FrankaKitchen {self.task_name}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        ):
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

        sr = np.asarray(ep_success_levels, dtype=np.float64)
        mean_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        std_lat = float(np.std(latencies_ms)) if latencies_ms else 0.0
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
            "n_infer_episodes": int(n_eps),
        }
        out["trade_off"] = (
            float(out["success_rate_k4"] / mean_lat) if mean_lat > 1e-9 else 0.0
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

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]

        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data["sim_video_eval"] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data
