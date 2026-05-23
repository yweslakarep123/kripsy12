from typing import Dict, List, Optional
import os
import pathlib
import torch
import numpy as np
import copy
from omegaconf import OmegaConf
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.replay_buffer import ReplayBuffer
from flow_policy_3d.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.dataset.base_dataset import BaseDataset


def _resolve_zarr_path(zarr_path: str) -> str:
    """Resolve dataset path: absolute paths unchanged; relative paths are relative to the
    FlowPolicy package root (directory containing ``train.py`` and ``flow_policy_3d/``),
    not the shell's original cwd (Hydra ``to_absolute_path`` uses that and breaks when
    launching ``python FlowPolicy/train.py`` from the repo root).
    """
    zarr_path = str(zarr_path).strip()
    if not zarr_path:
        return zarr_path
    if os.path.isabs(zarr_path):
        return os.path.normpath(os.path.expanduser(zarr_path))
    pkg_root = pathlib.Path(__file__).resolve().parents[2]
    return str((pkg_root / zarr_path).resolve())


class KitchenDataset(BaseDataset):
    """Zarr dataset: ``state``, ``action``, dan opsional ``point_cloud``.

    Mode state-only (Franka Kitchen): hanya ``state`` + ``action``; ``agent_pos`` = ``state``.
    """

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        train_episode_indices: Optional[List[int]] = None,
        val_episode_indices: Optional[List[int]] = None,
        preprocessing_profile: str = "minimal",
        obs_noise_std: Optional[float] = None,
    ):
        super().__init__()
        if zarr_path is None:
            raise ValueError(
                "KitchenDataset: zarr_path tidak boleh None. "
                "Atur di YAML task (dataset.zarr_path) atau CLI, mis. "
                "task.dataset.zarr_path=/path/ke/franka_kitchen_sequential4_expert.zarr"
            )
        zarr_path = str(zarr_path).strip()
        if not zarr_path:
            raise ValueError(
                "KitchenDataset: zarr_path kosong. Biasanya karena override Hydra/shell "
                "tanpa nilai (mis. variabel lingkungan kosong). "
                "Contoh: task.dataset.zarr_path=data/franka_kitchen_sequential4_expert.zarr "
                "atau path absolut ke dataset .zarr."
            )
        zarr_path = _resolve_zarr_path(zarr_path)
        if not os.path.isdir(zarr_path):
            raise ValueError(
                "KitchenDataset: direktori zarr tidak ditemukan:\n"
                f"  {zarr_path}\n"
                "Path relatif dihitung dari folder paket FlowPolicy (tempat train.py). "
                "Gunakan path absolut jika dataset berada di lokasi lain."
            )
        import zarr as _zarr

        root = _zarr.open(zarr_path, mode="r")
        keys = ["state", "action"]
        if "point_cloud" in root["data"]:
            keys.append("point_cloud")
        self._has_point_cloud = "point_cloud" in keys
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)
        n_eps = self.replay_buffer.n_episodes
        self._explicit_val_indices: Optional[List[int]] = None
        profile = (preprocessing_profile or "minimal").lower()
        if profile == "standard":
            self._obs_noise_std = 0.01 if obs_noise_std is None else float(obs_noise_std)
        elif profile == "minimal":
            self._obs_noise_std = float(obs_noise_std) if obs_noise_std is not None else 0.0
        else:
            raise ValueError(
                "KitchenDataset: preprocessing_profile harus 'standard' atau 'minimal', "
                f"dapat {preprocessing_profile!r}"
            )

        if train_episode_indices is not None:
            if OmegaConf.is_config(train_episode_indices):
                train_episode_indices = list(
                    OmegaConf.to_container(train_episode_indices, resolve=True)
                )
            if OmegaConf.is_config(val_episode_indices):
                val_episode_indices = list(
                    OmegaConf.to_container(val_episode_indices, resolve=True)
                )
            if val_episode_indices is None:
                raise ValueError(
                    "KitchenDataset: val_episode_indices wajib diisi jika "
                    "train_episode_indices dipakai (split CV eksplisit)."
                )
            train_mask = np.zeros(n_eps, dtype=bool)
            for i in train_episode_indices:
                if i < 0 or i >= n_eps:
                    raise ValueError(
                        f"KitchenDataset: indeks episode latih tidak valid: {i} "
                        f"(jumlah episode dalam zarr: {n_eps})"
                    )
                train_mask[i] = True
            for i in val_episode_indices:
                if i < 0 or i >= n_eps:
                    raise ValueError(
                        f"KitchenDataset: indeks episode validasi tidak valid: {i}"
                    )
                if train_mask[i]:
                    raise ValueError(
                        "KitchenDataset: episode tidak boleh masuk train dan val sekaligus "
                        f"(episode {i})."
                    )
            self._explicit_val_indices = list(val_episode_indices)
        else:
            val_mask = get_val_mask(
                n_episodes=n_eps,
                val_ratio=val_ratio,
                seed=seed,
            )
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask, max_n=max_train_episodes, seed=seed
            )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        if self._explicit_val_indices is not None:
            val_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            val_mask[self._explicit_val_indices] = True
        else:
            val_mask = ~self.train_mask
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
        )
        val_set.train_mask = val_mask
        val_set._obs_noise_std = 0.0
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
        }
        if self._has_point_cloud:
            data["point_cloud"] = self.replay_buffer["point_cloud"]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][:,].astype(np.float32)
        obs = {"agent_pos": agent_pos}
        if self._has_point_cloud:
            obs["point_cloud"] = sample["point_cloud"][:,].astype(np.float32)
        data = {"obs": obs, "action": sample["action"].astype(np.float32)}
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        if self._obs_noise_std > 0:
            std = self._obs_noise_std
            ap = torch_data["obs"]["agent_pos"]
            torch_data["obs"]["agent_pos"] = ap + torch.randn_like(ap) * std
            if "point_cloud" in torch_data["obs"]:
                pc = torch_data["obs"]["point_cloud"]
                torch_data["obs"]["point_cloud"] = pc.clone()
                torch_data["obs"]["point_cloud"][..., :3] = (
                    pc[..., :3] + torch.randn_like(pc[..., :3]) * std
                )
        return torch_data
