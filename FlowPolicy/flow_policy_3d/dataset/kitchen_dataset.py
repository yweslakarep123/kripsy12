from typing import Dict
import os
import pathlib
import torch
import numpy as np
import copy
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
    """Zarr dataset with keys state, action, point_cloud (same layout as MetaworldDataset)."""

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
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
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["state", "action", "point_cloud"]
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
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
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][...,:],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][:,].astype(np.float32)
        point_cloud = sample["point_cloud"][:,].astype(np.float32)

        data = {
            "obs": {
                "point_cloud": point_cloud,
                "agent_pos": agent_pos,
            },
            "action": sample["action"].astype(np.float32),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
