from typing import Dict, Optional, Set
import json
import torch
import numpy as np
import copy
import pathlib
from tqdm import tqdm
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.replay_buffer import ReplayBuffer
from flow_policy_3d.common.sampler import SequenceSampler, get_val_mask
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.dataset.base_dataset import BaseDataset
from flow_policy_3d.env.kitchen.kitchen_util import parse_mjl_logs


def _load_episode_split(path: str) -> Dict[str, Set[int]]:
    with open(path) as f:
        raw = json.load(f)
    split = raw.get("split", raw)
    return {
        "train": set(int(i) for i in split["train_episodes"]),
        "val": set(int(i) for i in split["val_episodes"]),
        "test": set(int(i) for i in split.get("test_episodes", [])),
    }


def _timestep_mask_for_episodes(replay_buffer, episode_mask: np.ndarray) -> np.ndarray:
    episode_ends = replay_buffer.episode_ends[:]
    n_eps = len(episode_mask)
    if len(episode_ends) == 0:
        return np.zeros(0, dtype=bool)
    total = int(episode_ends[-1])
    step_mask = np.zeros(total, dtype=bool)
    for i in range(n_eps):
        if not episode_mask[i]:
            continue
        start = 0 if i == 0 else int(episode_ends[i - 1])
        end = int(episode_ends[i])
        step_mask[start:end] = True
    return step_mask


class KitchenMjlLowdimDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=True,
        robot_noise_ratio=0.0,
        seed=42,
        val_ratio=0.0,
        episode_split_path: Optional[str] = None,
    ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array(
            [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.005,
                0.005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.005,
                0.005,
                0.005,
                0.1,
                0.1,
                0.1,
                0.005,
                0.005,
                0.005,
                0.1,
                0.1,
                0.1,
                0.005,
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(seed=seed)

        data_directory = pathlib.Path(dataset_dir)
        mjl_paths = sorted(data_directory.glob("*/*.mjl"))
        if not mjl_paths:
            raise FileNotFoundError(
                f"Tidak ada file */*.mjl di {data_directory.resolve()}"
            )

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        n_parse_errors = 0
        for i, mjl_path in enumerate(tqdm(mjl_paths, desc="Load kitchen MJL")):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data["qpos"].astype(np.float32)
                obs = np.concatenate(
                    [
                        qpos[:, :9],
                        qpos[:, -21:],
                        np.zeros((len(qpos), 30), dtype=np.float32),
                    ],
                    axis=-1,
                )
                if robot_noise_ratio > 0:
                    noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                        low=-1.0, high=1.0, size=(obs.shape[0], 30)
                    )
                    obs[:, :30] += noise
                episode = {
                    "obs": obs,
                    "action": data["ctrl"].astype(np.float32),
                }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                n_parse_errors += 1
                print(i, e)

        n_loaded = self.replay_buffer.n_episodes
        if n_loaded == 0:
            raise FileNotFoundError(
                f"Tidak ada episode MJL yang dimuat dari {data_directory.resolve()}. "
                f"Glob menemukan {len(mjl_paths)} file, {n_parse_errors} gagal parse."
            )
        if n_loaded != len(mjl_paths):
            raise RuntimeError(
                f"Hanya {n_loaded}/{len(mjl_paths)} episode berhasil dimuat."
            )

        self.episode_split_path = episode_split_path
        if episode_split_path:
            split = _load_episode_split(episode_split_path)
            n_eps = n_loaded
            for name, indices in split.items():
                bad = [i for i in indices if i < 0 or i >= n_eps]
                if bad:
                    raise ValueError(
                        f"Indeks {name} di luar [0, {n_eps}): {bad[:5]}..."
                    )
            overlap = (split["train"] & split["val"]) | (
                split["train"] & split["test"]
            ) | (split["val"] & split["test"])
            if overlap:
                raise ValueError(f"Overlap train/val/test: {sorted(overlap)[:10]}")

            train_mask = np.array(
                [i in split["train"] for i in range(n_eps)], dtype=bool
            )
            val_mask = np.array(
                [i in split["val"] for i in range(n_eps)], dtype=bool
            )
            if not train_mask.any() or not val_mask.any():
                raise ValueError(
                    f"Split kosong: train={train_mask.sum()}, val={val_mask.sum()}"
                )
        else:
            val_mask = get_val_mask(
                n_episodes=n_loaded,
                val_ratio=val_ratio,
                seed=seed,
            )
            train_mask = ~val_mask

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

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
            episode_mask=self.val_mask,
        )
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        step_mask = _timestep_mask_for_episodes(self.replay_buffer, self.train_mask)
        data = {
            "obs": self.replay_buffer["obs"][step_mask],
            "action": self.replay_buffer["action"][step_mask],
        }
        if "range_eps" not in kwargs:
            kwargs["range_eps"] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(sample, torch.from_numpy)
        return torch_data
