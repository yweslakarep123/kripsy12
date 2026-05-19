"""Adapter `FlowMLP`-compatible yang membungkus `ConditionalUnet1D` FlowPolicy.

ReinFlow's `NoisyFlowMLP` mengharapkan kelas policy (FlowMLP) dengan:
- atribut: `horizon_steps`, `action_dim`, `act_dim_total`, `time_dim`, `cond_enc_dim`
- method: `forward(action, time, cond, output_embedding=False, **kwargs)`
           -> velocity  OR  (velocity, time_emb, cond_emb) jika output_embedding=True
- method: `sample_action(cond, inference_steps, clip_intermediate_actions, act_range, z=None, save_chains=False)`

Adapter ini menggantikan FlowMLP dengan ConditionalUnet1D + global_cond berbasis state
yang sudah DI-ENCODE di env-wrapper (lihat `finetune_flowpolicy/envs/encoder_wrapper.py`).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

import finetune_flowpolicy.paths  # noqa: F401  side-effect: setup sys.path
from flow_policy_3d.model.flow.conditional_unet1d import ConditionalUnet1D


class FlowPolicyVelocityAdapter(nn.Module):
    """Drop-in pengganti `FlowMLP` untuk ReinFlow PPO yang membungkus ConditionalUnet1D.

    Catatan: encoder point-cloud (FlowPolicyEncoder) di-handle terpisah oleh env wrapper.
    Observasi yang masuk ke adapter ini sudah berupa flat feature `cond["state"]`.
    """

    # konstanta yang dipakai FlowPolicy.predict_action saat memanggil UNet
    TIME_SCALE: float = 99.0

    def __init__(
        self,
        *,
        action_dim: int,
        horizon_steps: int,
        cond_steps: int,
        obs_feature_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: Sequence[int] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = "film",
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
    ) -> None:
        super().__init__()

        self.action_dim = int(action_dim)
        self.horizon_steps = int(horizon_steps)
        self.act_dim_total = self.action_dim * self.horizon_steps
        self.cond_steps = int(cond_steps)
        self.obs_feature_dim = int(obs_feature_dim)
        self.cond_enc_dim = self.obs_feature_dim * self.cond_steps
        # NoisyFlowMLP mem-konkat time_emb (atau time_emb_explore) + cond_emb -> noise_feature
        # untuk kompatibilitas, time_dim = diffusion_step_embed_dim
        self.time_dim = int(diffusion_step_embed_dim)

        self.unet = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=self.cond_enc_dim,
            diffusion_step_embed_dim=self.time_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

    # ------------------------------------------------------------------
    # Loading checkpoint pretrained
    # ------------------------------------------------------------------

    def load_unet_state_dict(self, unet_sd: dict, strict: bool = True) -> None:
        """Muat state_dict ConditionalUnet1D dari pretrained FlowPolicy checkpoint."""
        missing, unexpected = self.unet.load_state_dict(unet_sd, strict=strict)
        if missing or unexpected:
            raise RuntimeError(
                f"load_unet_state_dict mismatch: missing={missing}, unexpected={unexpected}"
            )

    # ------------------------------------------------------------------
    # FlowMLP-compatible interface
    # ------------------------------------------------------------------

    def _extract_global_cond(self, cond) -> Tensor:
        """Konversi cond `dict{"state": (B, To, Do)}` atau Tensor `(B, To, Do)` -> flat (B, To*Do)."""
        if isinstance(cond, dict):
            state = cond["state"]
        else:
            state = cond
        B = state.shape[0]
        return state.reshape(B, -1)

    def forward(
        self,
        action: Tensor,
        time: Tensor | float | int,
        cond,
        output_embedding: bool = False,
        **kwargs,
    ):
        """Predict velocity. Interface persis seperti FlowMLP.

        Args:
            action: (B, Ta, Da) di ruang ternormalisasi [-1, 1].
            time:   (B,) atau float di [0, 1) (waktu Rectified Flow).
            cond:   dict{"state": (B, To, Do)} atau Tensor (B, To, Do).
            output_embedding: jika True, kembalikan juga (time_emb, cond_emb) untuk NoisyFlowMLP.

        Returns:
            velocity: (B, Ta, Da)   [+ (time_emb (B, time_dim), cond_emb (B, cond_enc_dim))]
        """
        if action.dim() != 3:
            raise ValueError(f"action harus (B, Ta, Da), dapat shape {tuple(action.shape)}")
        B = action.shape[0]
        device = action.device

        global_cond = self._extract_global_cond(cond).to(device=device, dtype=action.dtype)

        # Konversi time RF (float [0,1)) -> tensor (B,) di device action.
        if isinstance(time, (int, float)):
            t_tensor = torch.full((B,), float(time), device=device, dtype=action.dtype)
        else:
            t_tensor = time.to(device=device, dtype=action.dtype)
            if t_tensor.dim() == 0:
                t_tensor = t_tensor[None].expand(B)
            elif t_tensor.dim() == 1 and t_tensor.shape[0] != B:
                t_tensor = t_tensor.expand(B)

        # FlowPolicy memanggil UNet dengan timestep di-skala 99x (konsisten dengan pretraining).
        t_scaled = t_tensor * self.TIME_SCALE

        vel = self.unet(action, t_scaled, local_cond=None, global_cond=global_cond)

        if output_embedding:
            # NoisyFlowMLP butuh time_emb (shape (B, time_dim)) dan cond_emb (shape (B, cond_enc_dim)).
            time_emb = self.unet.diffusion_step_encoder(t_scaled)  # (B, time_dim)
            cond_emb = global_cond                                  # (B, cond_enc_dim)
            return vel, time_emb, cond_emb
        return vel

    @torch.no_grad()
    def sample_action(
        self,
        cond,
        inference_steps: int,
        clip_intermediate_actions: bool,
        act_range: List[float],
        z: Optional[Tensor] = None,
        save_chains: bool = False,
    ):
        """Euler ODE sampling identik dengan FlowMLP.sample_action.

        Args:
            cond: dict{"state": ...} atau tensor sudah encoded.
            inference_steps: K (jumlah Euler step). Untuk fine-tuning K=1 secara default.
            clip_intermediate_actions: jika True, clamp ke act_range setiap step.
            act_range: [act_min, act_max].
            z: initial noise (B, Ta, Da). Jika None, di-sample N(0, I).
        """
        state = cond["state"] if isinstance(cond, dict) else cond
        B = state.shape[0]
        device = state.device

        if z is None:
            x_hat = torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        else:
            x_hat = z.clone()

        if save_chains:
            x_chain = torch.zeros(
                (B, inference_steps + 1, self.horizon_steps, self.action_dim),
                device=device,
            )
            x_chain[:, 0] = x_hat

        dt = 1.0 / float(inference_steps)
        # waktu untuk tiap step: 0, 1/K, ..., (K-1)/K (mengikuti FlowMLP.sample_action)
        steps = torch.linspace(0.0, 1.0 - dt, inference_steps, device=device).repeat(B, 1)
        dt_tensor = dt * torch.ones_like(x_hat, device=device)

        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.forward(x_hat, t, cond)
            x_hat = x_hat + vt * dt_tensor
            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(act_range[0], act_range[1])
            if save_chains:
                x_chain[:, i + 1] = x_hat

        if save_chains:
            return x_hat, x_chain
        return x_hat
