"""Wrapper untuk FrankaKitchenPointCloudEnv yang:
1) Pre-encode dict obs (point_cloud + agent_pos) menjadi flat ``{"state": (D,)}``
   memakai ``FlowPolicyEncoder`` (FROZEN) + ``LinearNormalizer`` dari checkpoint.
2) Unnormalize action [-1, 1] (output policy) menjadi ruang aksi raw env via normalizer.
3) Mengubah API reset Gymnasium (obs, info) menjadi gym-style (obs saja) supaya
   kompatibel dengan ``ReinFlow MultiStep`` wrapper.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import torch
from gym import spaces

import finetune_flowpolicy.paths  # noqa: F401  side-effect: setup sys.path
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.model.vision.pointnet_extractor import FlowPolicyEncoder


class PreEncodeObsWrapper(gym.Wrapper):
    """Pre-encode point-cloud + agent_pos -> flat feature; unnormalize aksi.

    Output observation::
        {"state": np.ndarray shape (obs_feature_dim,), dtype=float32}

    Encoder & normalizer di-FREEZE (eval mode, requires_grad_=False) supaya
    konsisten dengan pretrained FlowPolicy.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        encoder: FlowPolicyEncoder,
        normalizer: LinearNormalizer,
        device: str = "cpu",
        use_pc_color: bool = False,
        obs_feature_dim: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self._device = torch.device(device)
        self._encoder = encoder.to(self._device).eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)
        self._normalizer = normalizer.to(self._device)
        for p in self._normalizer.parameters():
            p.requires_grad_(False)
        self._use_pc_color = bool(use_pc_color)

        if obs_feature_dim is None:
            obs_feature_dim = int(self._encoder.output_shape())
        self._obs_feature_dim = int(obs_feature_dim)

        # Spaces
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._obs_feature_dim,),
                    dtype=np.float32,
                )
            }
        )
        # Action policy bekerja di ruang ternormalisasi [-1, 1] per dim aksi.
        env_act_dim = int(np.prod(env.action_space.shape))
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(env_act_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_obs(self, raw_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Encode dict obs -> {"state": (obs_feature_dim,) float32}."""
        # ke tensor di device encoder, tambah batch-dim
        pc = torch.as_tensor(raw_obs["point_cloud"], dtype=torch.float32, device=self._device).unsqueeze(0)
        ap = torch.as_tensor(raw_obs["agent_pos"], dtype=torch.float32, device=self._device).unsqueeze(0)
        # normalisasi via LinearNormalizer (key-based)
        normed = self._normalizer.normalize({"point_cloud": pc, "agent_pos": ap})
        if not self._use_pc_color:
            normed["point_cloud"] = normed["point_cloud"][..., :3]
        feat = self._encoder(normed)  # (1, obs_feature_dim)
        return {"state": feat.squeeze(0).detach().cpu().numpy().astype(np.float32)}

    @torch.no_grad()
    def _unnormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        """Action [-1,1] -> ruang env asli."""
        act = torch.as_tensor(action_norm, dtype=torch.float32, device=self._device)
        unnorm = self._normalizer["action"].unnormalize(act)
        return unnorm.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Dict[str, np.ndarray]:
        """Return obs only (gym-style) supaya kompatibel dengan MultiStep.

        Menerima kwargs seed/options/return_info dari MultiStep & AsyncVectorEnv.
        """
        return_info = bool(kwargs.pop("return_info", False))
        # FrankaKitchenPointCloudEnv.reset() return (obs, info) -- 2-tuple
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            raw_obs, info = result
        else:
            raw_obs, info = result, {}
        encoded = self._encode_obs(raw_obs)
        if return_info:
            return encoded, info
        return encoded

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """Gym-style 4-tuple step (MultiStep akan ubah jadi 5-tuple)."""
        raw_action = self._unnormalize_action(np.asarray(action, dtype=np.float32))
        raw_obs, reward, done, info = self.env.step(raw_action)
        encoded = self._encode_obs(raw_obs)
        return encoded, float(reward), bool(done), info

    def seed(self, seed: Optional[int] = None) -> None:
        # FrankaKitchen env ignore seed di luar reset; tetap teruskan agar kompatibel.
        if hasattr(self.env, "seed"):
            try:
                self.env.seed(seed)
            except Exception:
                pass

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
        # encoder & normalizer cukup di-detach (tidak ada resource OS yang dipegang)

    @property
    def obs_feature_dim(self) -> int:
        return self._obs_feature_dim
