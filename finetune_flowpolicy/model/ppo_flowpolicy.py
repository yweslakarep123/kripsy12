"""PPOFlowAdapter: subclass `PPOFlow` (ReinFlow) yang menggunakan
`FlowPolicyVelocityAdapter` (ConditionalUnet1D) sebagai network velocity, alih-alih
`FlowMLP`. Loader checkpoint juga di-override agar membaca format FlowPolicy
(``state_dicts.ema_model``) bukan format ReinFlow (``model.network.*``).
"""
from __future__ import annotations

import copy
import logging
from typing import Any

import torch

import finetune_flowpolicy.paths  # noqa: F401  side-effect: setup sys.path
from finetune_flowpolicy.adapters.flowpolicy_velocity_adapter import FlowPolicyVelocityAdapter
from finetune_flowpolicy.utils.ckpt_io import (
    load_payload,
    select_full_state_dict,
)
from model.flow.ft_ppo.ppoflow import PPOFlow
from model.flow.mlp_flow import NoisyFlowMLP

log = logging.getLogger(__name__)


class PPOFlowAdapter(PPOFlow):
    """Override `load_policy` agar load dari checkpoint format FlowPolicy.

    Catatan tentang `init_actor_ft`: tidak perlu di-override karena `NoisyFlowMLP`
    hanya butuh atribut `policy.cond_enc_dim`, `policy.time_dim`,
    `policy.act_dim_total`, `policy.horizon_steps`, `policy.action_dim`,
    dan method `policy.forward(action, time, cond, output_embedding=True)` --
    semua sudah disediakan `FlowPolicyVelocityAdapter`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # `policy` arg di parent dikirim sebagai instance `FlowPolicyVelocityAdapter`
        # (lihat cfg `model.policy._target_=FlowPolicyVelocityAdapter`).
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Override checkpoint loader
    # ------------------------------------------------------------------

    def load_policy(self, network_path: str, use_ema: bool = False) -> None:
        """Muat pretrained ConditionalUnet1D dari checkpoint FlowPolicy.

        Args:
            network_path: path ``*.ckpt`` FlowPolicy (mis. ``latest-001.ckpt``).
            use_ema: ignored signature compatibility - selalu pakai EMA model
                jika tersedia (lebih bagus untuk init RL).
        """
        if not network_path:
            log.warning("No actor policy path provided. Skipping pretrained load.")
            return

        log.info("Loading FlowPolicy pretrained checkpoint from %s", network_path)
        payload = load_payload(network_path, map_location=str(self.device))
        full_sd = select_full_state_dict(payload, use_ema=True)
        unet_sd = {
            k[len("model."):]: v
            for k, v in full_sd.items()
            if k.startswith("model.")
        }
        if not unet_sd:
            raise RuntimeError(
                f"Tidak menemukan key dengan prefix 'model.' di checkpoint {network_path}"
            )

        # Load ke actor_old (FlowPolicyVelocityAdapter -- instance dari `policy` cfg)
        actor_old: FlowPolicyVelocityAdapter = self.actor_old  # type: ignore[assignment]
        actor_old.load_unet_state_dict(unet_sd, strict=True)
        log.info(
            "Loaded EMA ConditionalUnet1D weights into actor_old (params=%.2fM)",
            sum(p.numel() for p in actor_old.parameters()) / 1e6,
        )

    # ------------------------------------------------------------------
    # Override init_actor_ft: identik dengan PPOFlow tapi pastikan policy_copy
    # tetap FlowPolicyVelocityAdapter (deepcopy bisa berfungsi langsung di nn.Module).
    # ------------------------------------------------------------------

    def init_actor_ft(self, policy_copy: FlowPolicyVelocityAdapter) -> None:
        """NoisyFlowMLP membungkus policy_copy. Atribut yang dibutuhkan oleh
        NoisyFlowMLP sudah disediakan oleh FlowPolicyVelocityAdapter.
        """
        self.actor_ft = NoisyFlowMLP(
            policy=policy_copy,
            denoising_steps=self.inference_steps,
            learn_explore_noise_from=self.inference_steps - self.ft_denoising_steps,
            inital_noise_scheduler_type=self.noise_scheduler_type,
            min_logprob_denoising_std=self.min_logprob_denoising_std,
            max_logprob_denoising_std=self.max_logprob_denoising_std,
            learn_explore_time_embedding=self.learn_explore_time_embedding,
            time_dim_explore=self.time_dim_explore,
            use_time_independent_noise=self.use_time_independent_noise,
            device=self.device,
            noise_hidden_dims=self.noise_hidden_dims,
            activation_type=self.explore_net_activation_type,
        )
