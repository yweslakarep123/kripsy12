"""GPU-resident batched loader for small offline datasets (e.g. Kitchen MJL).

Preloads all samples onto VRAM once to avoid DataLoader worker / H2D overhead.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class GpuCachedBatchedLoader:
    """Iterable loader: materialize dataset on GPU, yield shuffled mini-batches."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        device: torch.device | str = "cuda",
        seed: int = 0,
        show_progress: bool = True,
    ):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.n_samples = len(dataset)
        if self.n_samples == 0:
            self._data = {}
            self.cache_size_mb = 0.0
            return

        keys = list(dataset[0].keys())
        stacked: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
        it = range(self.n_samples)
        if show_progress:
            it = tqdm(it, desc="Cache dataset → GPU", leave=False)
        for i in it:
            sample = dataset[i]
            for k in keys:
                t = sample[k]
                if not isinstance(t, torch.Tensor):
                    t = torch.as_tensor(t)
                stacked[k].append(t)

        self._data: Dict[str, torch.Tensor] = {
            k: torch.stack(v, dim=0).to(self.device, non_blocking=True)
            for k, v in stacked.items()
        }
        nbytes = sum(t.numel() * t.element_size() for t in self._data.values())
        self.cache_size_mb = nbytes / (1024**2)

    def __len__(self) -> int:
        if self.n_samples == 0:
            return 0
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.n_samples == 0:
            return

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            order = torch.randperm(self.n_samples, generator=g)
        else:
            order = torch.arange(self.n_samples)

        for start in range(0, self.n_samples, self.batch_size):
            idx = order[start : start + self.batch_size]
            idx_dev = idx.to(self.device, non_blocking=True)
            yield {k: v.index_select(0, idx_dev) for k, v in self._data.items()}

    def set_epoch_seed(self, epoch: int) -> None:
        """Ganti seed shuffle per epoch agar urutan batch berbeda."""
        self.seed = int(epoch)
