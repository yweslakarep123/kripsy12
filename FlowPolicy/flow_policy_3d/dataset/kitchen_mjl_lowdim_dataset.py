from typing import Dict
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

class KitchenMjlLowdimDataset(BaseDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            robot_noise_ratio=0.0,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array([0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   ,
            0.1   , 0.005 , 0.005 , 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
            0.0005, 0.005 , 0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ,
            0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ], dtype=np.float32)
        rng = np.random.default_rng(seed=seed)

        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        mjl_paths = list(data_directory.glob('*/*.mjl'))
        n_parse_errors = 0
        for i, mjl_path in enumerate(tqdm(mjl_paths)):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data['qpos'].astype(np.float32)
                obs = np.concatenate([
                    qpos[:,:9],
                    qpos[:,-21:],
                    np.zeros((len(qpos),30),dtype=np.float32)
                ], axis=-1)
                if robot_noise_ratio > 0:
                    noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                        low=-1., high=1., size=(obs.shape[0], 30))
                    obs[:,:30] += noise
                episode = {
                    'obs': obs,
                    'action': data['ctrl'].astype(np.float32)
                }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                n_parse_errors += 1
                print(i, e)

        # #region agent log
        import json
        import time
        _dbg_payload = {
            "sessionId": "21f965",
            "hypothesisId": "B",
            "location": "kitchen_mjl_lowdim_dataset.py:__init__",
            "message": "dataset load summary",
            "data": {
                "dataset_dir": str(dataset_dir),
                "resolved_dir": str(data_directory.resolve()),
                "dir_exists": data_directory.is_dir(),
                "mjl_glob_count": len(mjl_paths),
                "n_episodes_loaded": int(self.replay_buffer.n_episodes),
                "n_parse_errors": n_parse_errors,
            },
            "timestamp": int(time.time() * 1000),
        }
        try:
            _dbg_log = pathlib.Path(__file__).resolve().parents[3] / ".cursor" / "debug-21f965.log"
            _dbg_log.parent.mkdir(parents=True, exist_ok=True)
            with open(_dbg_log, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(_dbg_payload) + "\n")
        except OSError:
            pass
        # #endregion

        if self.replay_buffer.n_episodes == 0:
            raise FileNotFoundError(
                f"Tidak ada episode MJL yang dimuat dari {data_directory.resolve()}. "
                f"Glob '*/*.mjl' menemukan {len(mjl_paths)} file, "
                f"{n_parse_errors} gagal parse. "
                "Pastikan dataset ada dan path benar (relatif ke FlowPolicy/: "
                "'data/kitchen/kitchen_demos_multitask')."
            )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

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
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action']
        }
        if 'range_eps' not in kwargs:
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(sample, torch.from_numpy)
        return torch_data
