"""Entrypoint Hydra untuk PPO fine-tuning FlowPolicy di Franka Kitchen.

Jalankan dari akar repositori::

    conda activate flowpolicy-kitchen
    python finetune_flowpolicy/scripts/run_ft_ppo.py \
        --config-path ../cfg --config-name ft_ppo_flowpolicy_kitchen \
        seed=101 device=cuda:0

Override Hydra umum yang berguna::

    env.n_envs=2 train.n_train_itr=2 train.n_steps=10 train.batch_size=20  # smoke test
    base_policy_path=/abs/path/ke/checkpoint.ckpt                              # ganti ckpt sumber
"""
from __future__ import annotations

import os
import pathlib
import sys

# Set MuJoCo headless renderer SEBELUM import gym/gymnasium.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_THIS = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Setup PYTHONPATH untuk FlowPolicy/ + ReinFlow/ (side-effect import).
from finetune_flowpolicy import paths as _paths  # noqa: F401

import hydra
from omegaconf import OmegaConf

# OmegaConf eval resolver (dipakai cfg.eval:'${obs_dim} * ${cond_steps}')
try:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception:
    pass


@hydra.main(
    version_base=None,
    config_path=str((_THIS.parent.parent / "cfg").resolve()),
    config_name="ft_ppo_flowpolicy_kitchen",
)
def main(cfg) -> None:
    # Resolve base_policy_path ke absolute (relative ke repo root) bila perlu.
    base_path = pathlib.Path(cfg.base_policy_path)
    if not base_path.is_absolute():
        base_path = (_REPO_ROOT / base_path).resolve()
    cfg.base_policy_path = str(base_path)

    # `cfg` adalah konfigurasi top-level yang juga dipakai child instantiations
    # (cfg.model, cfg.model.policy, dst), sehingga tidak bisa pakai hydra.utils.instantiate(cfg)
    # secara langsung (akan splat semua field cfg sebagai kwargs).
    # Solusi: ambil class via _target_, lalu panggil dengan cfg sebagai single positional arg.
    target_cls = hydra.utils.get_class(cfg._target_)
    agent = target_cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
