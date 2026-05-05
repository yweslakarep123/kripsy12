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
