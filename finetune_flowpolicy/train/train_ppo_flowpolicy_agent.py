"""Trainer PPO untuk FlowPolicy di Franka Kitchen.

Subclass `TrainPPOFlowAgent` (ReinFlow) yang:
1. Mengganti pembuatan vec env dari `make_async` (gym registry) ke factory
   FrankaKitchen kustom kita lewat *monkey-patch* lokal (tanpa mengedit ReinFlow).
2. Menambah hook ``save_model`` untuk juga menyimpan ``*.ckpt`` format FlowPolicy
   sehingga ``infer_kitchen.py`` bisa langsung dipakai untuk evaluasi tanpa adaptasi.
"""
from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

import numpy as np
import torch

import finetune_flowpolicy.paths  # noqa: F401  side-effect: setup sys.path
from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent
from finetune_flowpolicy.adapters.flowpolicy_velocity_adapter import FlowPolicyVelocityAdapter
from finetune_flowpolicy.envs.franka_kitchen_vec import make_franka_kitchen_vec_env
from finetune_flowpolicy.utils.ckpt_io import export_to_flowpolicy_ckpt

log = logging.getLogger(__name__)


class TrainPPOFlowPolicyAgent(TrainPPOFlowAgent):
    """PPO trainer untuk FlowPolicy + Franka Kitchen."""

    def __init__(self, cfg: Any) -> None:
        # ------------------------------------------------------------------
        # Monkey-patch `make_async` di SEMUA module yang sudah mengimpornya
        # by-reference. ReinFlow TrainAgent melakukan `from env.gym_utils import make_async`,
        # sehingga patching module `env.gym_utils` saja tidak cukup -- binding di
        # `agent.finetune.reinflow.train_agent` sudah resolved sebelum patch.
        # Patch dipasang SEBELUM super().__init__(), dicabut setelahnya.
        # ------------------------------------------------------------------
        import env.gym_utils as _gym_utils_mod  # type: ignore[import-not-found]
        import agent.finetune.reinflow.train_agent as _train_agent_mod  # type: ignore[import-not-found]

        env_cfg = cfg.env
        pretrained_ckpt = cfg.get("base_policy_path", None)
        if pretrained_ckpt is None:
            raise ValueError("cfg.base_policy_path harus diisi (checkpoint FlowPolicy).")

        # `cfg` mungkin dimodifikasi oleh super (mis. env.specific) -- snapshot dulu.
        _cfg_snapshot = cfg

        def _patched_make_async(env_name, num_envs=1, asynchronous=True, **kwargs):
            if kwargs.get("env_type", None) == "franka_kitchen_pc":
                env_cfg_local = _cfg_snapshot.env
                async_override = bool(env_cfg_local.get("asynchronous", False))
                venv, obs_feature_dim, action_dim, _ = make_franka_kitchen_vec_env(
                    n_envs=num_envs,
                    pretrained_ckpt=pretrained_ckpt,
                    n_obs_steps=int(_cfg_snapshot.cond_steps),
                    n_action_steps=int(_cfg_snapshot.act_steps),
                    max_episode_steps=int(env_cfg_local.max_episode_steps),
                    tasks_to_complete=list(env_cfg_local.tasks_to_complete)
                    if env_cfg_local.get("tasks_to_complete", None) is not None
                    else None,
                    task_completion_order=list(env_cfg_local.task_completion_order)
                    if env_cfg_local.get("task_completion_order", None) is not None
                    else None,
                    terminate_on_tasks_completed=bool(
                        env_cfg_local.get("terminate_on_tasks_completed", True)
                    ),
                    num_points=int(env_cfg_local.get("num_points", 512)),
                    use_point_crop=bool(env_cfg_local.get("use_point_crop", True)),
                    asynchronous=async_override,
                    encoder_device=str(env_cfg_local.get("encoder_device", "cpu")),
                    seed_base=int(_cfg_snapshot.get("seed", 42)),
                )
                if int(_cfg_snapshot.obs_dim) != int(obs_feature_dim):
                    log.warning(
                        "cfg.obs_dim=%d != obs_feature_dim (encoder)=%d. Memakai nilai encoder.",
                        int(_cfg_snapshot.obs_dim),
                        obs_feature_dim,
                    )
                return venv
            return _orig_make_async_module(env_name, num_envs, asynchronous, **kwargs)

        _orig_make_async_module = _gym_utils_mod.make_async
        _orig_make_async_train = getattr(_train_agent_mod, "make_async", None)
        _gym_utils_mod.make_async = _patched_make_async
        _train_agent_mod.make_async = _patched_make_async
        try:
            super().__init__(cfg)
        finally:
            _gym_utils_mod.make_async = _orig_make_async_module
            if _orig_make_async_train is not None:
                _train_agent_mod.make_async = _orig_make_async_train

        # ------------------------------------------------------------------
        # Simpan path checkpoint sumber untuk exporter
        # ------------------------------------------------------------------
        self._pretrained_ckpt_path = pathlib.Path(pretrained_ckpt).resolve()

        # ------------------------------------------------------------------
        # Relax `initial_ratio_error_threshold` -- default ReinFlow=1e-6 terlalu ketat
        # untuk pipeline kita (encoder dijalankan di CPU di env worker, lalu feature
        # dikirim ke GPU; round-trip float32 menimbulkan jitter ~1e-6). Threshold 1e-4
        # masih cukup ketat untuk mendeteksi bug logprob recompute yang nyata.
        # ------------------------------------------------------------------
        self.initial_ratio_error_threshold = float(
            cfg.train.get("initial_ratio_error_threshold", 1e-4)
        )

    # ------------------------------------------------------------------
    # save_model: di samping ".pt" ReinFlow, juga emit ".ckpt" format FlowPolicy
    # ------------------------------------------------------------------

    def _emit_flowpolicy_ckpt(self, tag: str) -> None:
        """Tulis checkpoint format FlowPolicy ke ``checkpoint_dir/<tag>.ckpt``."""
        try:
            actor_ft = self.model.actor_ft.policy
            if not isinstance(actor_ft, FlowPolicyVelocityAdapter):
                log.warning(
                    "actor_ft.policy bukan FlowPolicyVelocityAdapter (%s); skip emit .ckpt",
                    type(actor_ft).__name__,
                )
                return
            unet_sd = {k: v.detach().cpu() for k, v in actor_ft.unet.state_dict().items()}
            out_path = pathlib.Path(self.checkpoint_dir) / f"{tag}.ckpt"
            export_to_flowpolicy_ckpt(
                src_ckpt_path=str(self._pretrained_ckpt_path),
                new_unet_sd=unet_sd,
                out_path=out_path,
            )
            log.info("Emitted FlowPolicy-format checkpoint -> %s", out_path)
        except Exception as e:
            log.exception("Gagal emit FlowPolicy .ckpt: %s", e)

    def save_model(self, only_save_policy_network: bool = False) -> None:  # type: ignore[override]
        """Override penuh. Memperbaiki bug `os.path.join(checkpoint_dir, save_path)` di
        ReinFlow `train_ppo_flow_agent.py` (parent menggabungkan path dua kali sehingga
        gagal di Linux), sekaligus juga menulis ``*.ckpt`` format FlowPolicy.
        """
        ckpt_dir = pathlib.Path(self.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        policy_network_state_dict = {
            "network." + k: v
            for k, v in self.model.actor_ft.policy.state_dict().items()
        }
        if only_save_policy_network:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "policy": policy_network_state_dict,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            }
        else:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "model": self.model.state_dict(),
                "policy": policy_network_state_dict,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            }

        # always save the last model for resume of training.
        last_pt = ckpt_dir / "last.pt"
        torch.save(data, str(last_pt))

        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            interm_pt = ckpt_dir / f"state_{self.itr}.pt"
            torch.save(data, str(interm_pt))
            log.info("Saved model at itr=%d to %s", self.itr, interm_pt)

        is_best = bool(self.is_best_so_far)
        if is_best:
            best_pt = ckpt_dir / "best.pt"
            torch.save(data, str(best_pt))
            log.info(
                "Saved best model (reward=%.3f) to %s",
                float(self.current_best_reward),
                best_pt,
            )
            self.is_best_so_far = False

        # tulis FlowPolicy-format ckpt: last + (opsional) best
        self._emit_flowpolicy_ckpt(tag="last_flowpolicy")
        if is_best:
            self._emit_flowpolicy_ckpt(tag="best_flowpolicy")
