"""Factory env vectorized Franka Kitchen (point-cloud) untuk ReinFlow PPO loop.

Stack wrapper::
    FrankaKitchenPointCloudEnv (dict obs, raw action)
        -> PreEncodeObsWrapper        (encode pc, unnormalize action)
        -> MultiStep (ReinFlow)       (stack n_obs_steps frames, loop n_action_steps)
        -> AsyncVectorEnv / SyncVectorEnv
"""
from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, List, Optional

import torch

import finetune_flowpolicy.paths  # noqa: F401  side-effect: setup sys.path
from env.gym_utils.async_vector_env import AsyncVectorEnv
from env.gym_utils.sync_vector_env import SyncVectorEnv
from env.gym_utils.wrapper.multi_step import MultiStep
from finetune_flowpolicy.envs.encoder_wrapper import PreEncodeObsWrapper
from finetune_flowpolicy.utils.ckpt_io import (
    load_payload,
    select_full_state_dict,
)
from flow_policy_3d.env.franka_kitchen.franka_kitchen_env import FrankaKitchenPointCloudEnv
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.model.vision.pointnet_extractor import FlowPolicyEncoder


def _build_encoder_from_cfg(policy_cfg: Any) -> FlowPolicyEncoder:
    """Re-instansiasi FlowPolicyEncoder dari cfg.policy yang tersimpan di checkpoint."""
    shape_meta = policy_cfg["shape_meta"]
    obs_meta = shape_meta["obs"]
    obs_space = {k: v["shape"] for k, v in obs_meta.items()}
    pc_cfg = copy.deepcopy(policy_cfg["pointcloud_encoder_cfg"])
    # `pointcloud_encoder_cfg.in_channels` di-overwrite di FlowPolicyEncoder.__init__
    enc = FlowPolicyEncoder(
        observation_space=obs_space,
        img_crop_shape=tuple(policy_cfg.get("crop_shape", (84, 84))),
        out_channel=int(policy_cfg["encoder_output_dim"]),
        pointcloud_encoder_cfg=pc_cfg,
        use_pc_color=bool(policy_cfg.get("use_pc_color", False)),
        pointnet_type=str(policy_cfg.get("pointnet_type", "mlp")),
    )
    return enc


def build_encoder_and_normalizer(
    pretrained_ckpt: str | pathlib.Path,
    *,
    use_ema: bool = True,
    device: str = "cpu",
):
    """Load encoder + normalizer dari checkpoint FlowPolicy, sudah FROZEN dan di-`eval()`.

    Returns:
        (encoder, normalizer, obs_feature_dim, cfg, use_pc_color)
    """
    payload = load_payload(pretrained_ckpt, map_location="cpu")
    full_sd = select_full_state_dict(payload, use_ema=use_ema)
    cfg = payload["cfg"]

    enc = _build_encoder_from_cfg(cfg["policy"])
    enc_sd = {k[len("obs_encoder."):]: v for k, v in full_sd.items() if k.startswith("obs_encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"encoder state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )
    enc = enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    normalizer = LinearNormalizer()
    norm_sd = {k[len("normalizer."):]: v for k, v in full_sd.items() if k.startswith("normalizer.")}
    normalizer.load_state_dict(norm_sd)
    normalizer = normalizer.to(device)
    for p in normalizer.parameters():
        p.requires_grad_(False)

    obs_feature_dim = int(enc.output_shape())
    return enc, normalizer, obs_feature_dim, cfg, bool(cfg["policy"].get("use_pc_color", False))


def make_franka_kitchen_vec_env(
    *,
    n_envs: int,
    pretrained_ckpt: str | pathlib.Path,
    n_obs_steps: int = 2,
    n_action_steps: int = 4,
    max_episode_steps: int = 280,
    tasks_to_complete: Optional[List[str]] = None,
    task_completion_order: Optional[List[str]] = None,
    terminate_on_tasks_completed: bool = True,
    num_points: int = 512,
    use_point_crop: bool = True,
    asynchronous: bool = False,
    encoder_device: str = "cpu",
    seed_base: int = 0,
):
    """Bangun vector env untuk PPO fine-tuning.

    Args:
        n_envs: jumlah env paralel.
        pretrained_ckpt: path checkpoint FlowPolicy (sumber encoder + normalizer).
        asynchronous: jika True pakai AsyncVectorEnv (proses terpisah), kalau False SyncVectorEnv.
        encoder_device: 'cpu' direkomendasikan kalau asynchronous=True (hindari ribut device GPU di worker).

    Returns:
        vec_env, obs_feature_dim, action_dim, cfg_snapshot
    """
    # Load encoder + normalizer SEKALI di proses utama, lalu di-deepcopy ke tiap worker.
    encoder, normalizer, obs_feature_dim, cfg, use_pc_color = build_encoder_and_normalizer(
        pretrained_ckpt, use_ema=True, device=encoder_device
    )
    # CPU agar aman di multiprocess (pickle GPU tensor bisa rumit di beberapa OS).
    encoder_cpu = encoder.cpu().eval()
    normalizer_cpu = normalizer.cpu()

    if tasks_to_complete is None:
        tasks_to_complete = ["microwave", "kettle", "slide cabinet", "light switch"]

    def env_fn(rank: int):
        # closure variables harus picklable; deepcopy encoder/normalizer untuk tiap worker
        local_encoder = copy.deepcopy(encoder_cpu)
        local_normalizer = copy.deepcopy(normalizer_cpu)
        local_pc_color = use_pc_color

        def _fn():
            base_env = FrankaKitchenPointCloudEnv(
                tasks_to_complete=tasks_to_complete,
                task_completion_order=task_completion_order,
                device="cpu",
                use_point_crop=use_point_crop,
                num_points=num_points,
                terminate_on_tasks_completed=terminate_on_tasks_completed,
                max_episode_steps=max_episode_steps,
            )
            wrapped = PreEncodeObsWrapper(
                base_env,
                encoder=local_encoder,
                normalizer=local_normalizer,
                device="cpu",
                use_pc_color=local_pc_color,
                obs_feature_dim=obs_feature_dim,
            )
            multi = MultiStep(
                wrapped,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
                reward_agg_method="sum",
                # NOTE: MultiStep.step selalu memanggil ``self.action.append(act)``
                # walaupun prev_action=False (lihat ReinFlow/env/gym_utils/wrapper/multi_step.py:176).
                # Aktifkan agar deque ``self.action`` ter-init di reset().
                prev_action=True,
                reset_within_step=False,
                pass_full_observations=False,
                verbose=False,
            )
            return multi

        return _fn

    env_fns = [env_fn(i) for i in range(n_envs)]
    if asynchronous:
        venv = AsyncVectorEnv(env_fns, dummy_env_fn=None, delay_init=False)
    else:
        venv = SyncVectorEnv(env_fns)
        # Workaround bug ReinFlow SyncVectorEnv.seed: memanggil `super().seed()`
        # tetapi gym 0.22's `VectorEnv` tidak punya method `seed` -> AttributeError.
        # Override seed instance untuk skip super-call dan langsung seed sub-envs.
        def _patched_seed(self, seed=None):  # type: ignore[no-redef]
            if seed is None:
                seed = [None] * self.num_envs
            elif isinstance(seed, int):
                seed = [seed + i for i in range(self.num_envs)]
            assert len(seed) == self.num_envs
            for env, single_seed in zip(self.envs, seed):
                try:
                    env.seed(single_seed)
                except Exception:
                    pass

        import types
        venv.seed = types.MethodType(_patched_seed, venv)

        # Tambahkan reset_arg / reset_one_arg (hanya didefinisikan oleh AsyncVectorEnv di ReinFlow).
        # ReinFlow TrainAgent.reset_env_all memanggil `venv.reset_arg(options_list=...)`.
        def _reset_arg(self, options_list, **kwargs):
            import numpy as _np
            obs_list = []
            for env, opts in zip(self.envs, options_list):
                opts = dict(opts) if opts else {}
                obs = env.reset(**opts)
                obs_list.append(obs)
            if isinstance(obs_list[0], dict):
                stacked = {}
                for k in obs_list[0].keys():
                    stacked[k] = _np.stack([o[k] for o in obs_list], axis=0)
                return stacked
            return _np.stack(obs_list, axis=0)

        def _reset_one_arg(self, env_ind, options=None):
            opts = dict(options) if options else {}
            return self.envs[env_ind].reset(**opts)

        venv.reset_arg = types.MethodType(_reset_arg, venv)
        venv.reset_one_arg = types.MethodType(_reset_one_arg, venv)

    # action_dim mengikuti FrankaKitchen (= 9)
    sample_act_space = venv.single_action_space if hasattr(venv, "single_action_space") else None
    if sample_act_space is not None and hasattr(sample_act_space, "shape"):
        action_dim = int(sample_act_space.shape[-1])
    else:
        action_dim = int(cfg["task"]["shape_meta"]["action"]["shape"][0])

    return venv, obs_feature_dim, action_dim, cfg
