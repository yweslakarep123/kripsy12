#!/usr/bin/env python3
"""
Fine-tuning FlowPolicy dengan RL online bergaya ReinFlow (ringkas):

  - Mean aksi = keluaran flow matching deterministik (``predict_action(..., deterministic=True)``).
  - Stokastisitas terlatih = diagonal Gaussian pada ruang aksi (``log_std_net`` pada fitur encoder),
    analog injeksi noise / reparameterisasi agar log-probabilitas terhitung untuk PPO.

Prasyarat: checkpoint BC (``TrainFlowPolicyWorkspace`` / ``latest.ckpt``).

Contoh:
  python train_reinflow_rl.py --bc-checkpoint runs/baseline_seed0_standard/checkpoints/latest.ckpt \\
    --output-dir runs/reinflow_rl_seed0_standard --seed 0
"""
from __future__ import annotations

import math
import argparse
import json
import os
import pathlib
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    _root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_root))
    os.chdir(str(_root))

from train import TrainFlowPolicyWorkspace  # noqa: E402
from flow_policy_3d.common.pytorch_util import dict_apply  # noqa: E402
from flow_policy_3d.policy.base_policy import BasePolicy  # noqa: E402


def _obs_feat(flow: torch.nn.Module, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Fitur global_cond sama seperti ``FlowPolicy.predict_action`` (tanpa SDE)."""
    nobs = flow.normalizer.normalize(obs_dict)
    if not flow.use_pc_color:
        nobs["point_cloud"] = nobs["point_cloud"][..., :3]
    B = obs_dict["point_cloud"].shape[0]
    To = flow.n_obs_steps
    this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
    nobs_features = flow.obs_encoder(this_nobs)
    if "cross_attention" in flow.condition_type:
        feat = nobs_features.reshape(B, flow.n_obs_steps, -1)
        feat = feat.reshape(B, -1)
    else:
        feat = nobs_features.reshape(B, -1)
    return feat


def _gaussian_log_prob(
    action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor:
    """Log π(a|s) diagonal; bentuk (B,)."""
    inv = (-0.5) * ((action - mean) / log_std.exp()) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
    return inv.sum(dim=tuple(range(1, inv.ndim)))


def _gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """``values`` panjang T+1 (termasuk bootstrap V(s_T)); reward/dones panjang T."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_v = float(values[t + 1])
        nonterm = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * next_v * nonterm - float(values[t])
        last_gae = delta + gamma * lam * nonterm * last_gae
        adv[t] = last_gae
    ret = adv + values[:T]
    return adv.astype(np.float32), ret.astype(np.float32)


class LogStdHead(nn.Module):
    def __init__(self, in_dim: int, n_flat: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Mish(),
            nn.Linear(256, n_flat),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat).clamp(-5.0, 0.5)


class ValueHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Mish(),
            nn.Linear(256, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat).squeeze(-1)


class ReinflowKitchenPolicy(BasePolicy):
    """
    Kebijakan rollout: ``a = μ_flow(s) + exp(log_std_φ(s)) ⊙ ε``.
    Eval: ``a = μ_flow(s)`` deterministik.
    """

    def __init__(self, flow: torch.nn.Module, log_std_head: LogStdHead, explore: bool):
        super().__init__()
        self.flow = flow
        self.log_std_head = log_std_head
        self.explore = explore

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.explore:
            return self.flow.predict_action(obs_dict, deterministic=True)
        feat = _obs_feat(self.flow, obs_dict)
        with torch.no_grad():
            mean = self.flow.predict_action(obs_dict, deterministic=True)["action"]
        log_std = self.log_std_head(feat).view_as(mean)
        std = log_std.exp().clamp_min(1e-4)
        eps = torch.randn_like(mean)
        action = mean + std * eps
        return {"action": action, "action_mean": mean, "log_std": log_std}

    def set_normalizer(self, normalizer):
        self.flow.set_normalizer(normalizer)


def _np_obs_to_torch(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        t = torch.from_numpy(np.asarray(v)).to(device=device, dtype=torch.float32)
        if t.ndim >= 2:
            out[k] = t.unsqueeze(0)
        else:
            out[k] = t.unsqueeze(0)
    return out


def _collect_rollout(
    env,
    policy: ReinflowKitchenPolicy,
    value_net: ValueHead,
    device: torch.device,
    n_steps: int,
) -> Dict[str, Any]:
    obs_list: List[Dict[str, np.ndarray]] = []
    act_list: List[np.ndarray] = []
    logp_list: List[float] = []
    rew_list: List[float] = []
    done_list: List[float] = []
    val_list: List[float] = []

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    policy.flow.eval()
    policy.log_std_head.train()
    value_net.train()

    for _ in range(n_steps):
        obs_t = _np_obs_to_torch(obs, device)
        with torch.no_grad():
            feat = _obs_feat(policy.flow, obs_t)
            mean = policy.flow.predict_action(obs_t, deterministic=True)["action"]
            log_std = policy.log_std_head(feat).view_as(mean)
            std = log_std.exp().clamp_min(1e-4)
            eps = torch.randn_like(mean)
            action = mean + std * eps
            logp = _gaussian_log_prob(action, mean, log_std).item()
            v = value_net(feat).item()

        np_act = action.squeeze(0).detach().cpu().numpy()
        next_obs, reward, done, info = env.step(np_act)
        done = bool(np.all(done))

        obs_list.append({k: np.copy(v) for k, v in obs.items()})
        act_list.append(np_act.astype(np.float32))
        logp_list.append(logp)
        rew_list.append(float(reward))
        done_list.append(1.0 if done else 0.0)
        val_list.append(v)

        obs = next_obs
        if done:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    with torch.no_grad():
        obs_t = _np_obs_to_torch(obs, device)
        feat = _obs_feat(policy.flow, obs_t)
        last_v = float(value_net(feat).item())

    return {
        "obs": obs_list,
        "actions": np.stack(act_list, axis=0),
        "log_probs": np.asarray(logp_list, dtype=np.float32),
        "rewards": np.asarray(rew_list, dtype=np.float32),
        "dones": np.asarray(done_list, dtype=np.float32),
        "values": np.asarray(val_list + [last_v], dtype=np.float32),
    }


def _ppo_update(
    buf: Dict[str, Any],
    policy: ReinflowKitchenPolicy,
    value_net: ValueHead,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    epochs: int,
    minibatch_size: int,
    *,
    finetune_flow: bool,
) -> Dict[str, float]:
    obs_all = buf["obs"]
    act = torch.as_tensor(buf["actions"], device=device)
    old_logp = torch.as_tensor(buf["log_probs"], device=device)
    adv, ret = _gae(
        buf["rewards"],
        buf["values"],
        buf["dones"],
        gamma=0.99,
        lam=0.95,
    )
    adv_t = torch.as_tensor(adv, device=device)
    ret_t = torch.as_tensor(ret, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    if finetune_flow:
        policy.flow.train()
    else:
        policy.flow.eval()

    T = act.shape[0]
    idx = np.arange(T)
    stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "n": 0}
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, T, minibatch_size):
            mb = idx[start : start + minibatch_size]
            if len(mb) == 0:
                continue
            b_obs = [obs_all[i] for i in mb]
            b_act = act[mb]
            b_old = old_logp[mb]
            b_adv = adv_t[mb]
            b_ret = ret_t[mb]

            feat_b = torch.cat(
                [_obs_feat(policy.flow, _np_obs_to_torch(o, device)) for o in b_obs],
                dim=0,
            )
            mean_b = torch.cat(
                [
                    policy.flow.predict_action(_np_obs_to_torch(o, device), deterministic=True)[
                        "action"
                    ]
                    for o in b_obs
                ],
                dim=0,
            )
            log_std_b = policy.log_std_head(feat_b).view_as(mean_b)
            new_logp = _gaussian_log_prob(b_act, mean_b, log_std_b)
            entropy = (0.5 * (1.0 + math.log(2 * math.pi)) + log_std_b).sum(
                dim=tuple(range(1, log_std_b.ndim))
            ).mean()
            values = value_net(feat_b)
            ratio = torch.exp(new_logp - b_old)
            clip_adv = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_adv
            pg_loss = -(torch.min(ratio * b_adv, clip_adv)).mean()
            v_loss = F.mse_loss(values, b_ret)
            loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            to_clip = list(policy.log_std_head.parameters()) + list(value_net.parameters())
            if any(p.requires_grad for p in policy.flow.parameters()):
                to_clip += [p for p in policy.flow.parameters() if p.requires_grad]
            nn.utils.clip_grad_norm_(to_clip, 1.0)
            optimizer.step()

            stats["pg"] += float(pg_loss.item())
            stats["v"] += float(v_loss.item())
            stats["ent"] += float(entropy.item())
            stats["n"] += 1
    n = max(stats["n"], 1)
    if finetune_flow:
        policy.flow.eval()
    return {k: stats[k] / n for k in ("pg", "v", "ent")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bc-checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-updates", type=int, default=50)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--ppo-epochs", type=int, default=6)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--finetune-flow", action="store_true")
    p.add_argument("--finetune-flow-lr", type=float, default=1e-6)
    p.add_argument("--n-infer-episodes", type=int, default=50)
    p.add_argument("--n-train-val-episodes", type=int, default=15)
    p.add_argument("--train-val-eval-seed-offset", type=int, default=31)
    p.add_argument("--skip-inference-videos", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir = pathlib.Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ws = TrainFlowPolicyWorkspace.create_from_checkpoint(args.bc_checkpoint)
    cfg = ws.cfg
    device = torch.device(cfg.training.device)
    flow = ws.ema_model if cfg.training.use_ema else ws.model
    flow.to(device)
    flow.eval()

    for param in flow.parameters():
        param.requires_grad = bool(args.finetune_flow)

    import hydra

    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(out_dir))
    env = runner.env

    spc = env.observation_space
    pc_shape = spc["point_cloud"].shape
    ap_shape = spc["agent_pos"].shape
    dummy_obs = {
        "point_cloud": torch.zeros((1,) + pc_shape, device=device, dtype=torch.float32),
        "agent_pos": torch.zeros((1,) + ap_shape, device=device, dtype=torch.float32),
    }
    with torch.no_grad():
        fdim = _obs_feat(flow, dummy_obs).shape[1]
    Ta = int(flow.n_action_steps)
    Da = int(flow.action_dim)

    log_std_head = LogStdHead(fdim, Ta * Da).to(device)
    value_net = ValueHead(fdim).to(device)

    params: List[Tuple[str, Any]] = [
        ("aux", list(log_std_head.parameters()) + list(value_net.parameters()), args.lr)
    ]
    if args.finetune_flow:
        params.append(("flow", list(flow.parameters()), args.finetune_flow_lr))
    opt_groups = [{"params": ps, "lr": lr} for _, ps, lr in params]
    optimizer = torch.optim.AdamW(opt_groups)

    pol_rollout = ReinflowKitchenPolicy(flow, log_std_head, explore=True)
    pol_eval = ReinflowKitchenPolicy(flow, log_std_head, explore=False)

    train_log: List[Dict[str, Any]] = []
    for it in range(int(args.total_updates)):
        buf = _collect_rollout(
            env,
            pol_rollout,
            value_net,
            device,
            int(args.rollout_steps),
        )
        st = _ppo_update(
            buf,
            pol_rollout,
            value_net,
            optimizer,
            device,
            float(args.clip_range),
            float(args.vf_coef),
            float(args.ent_coef),
            int(args.ppo_epochs),
            int(args.minibatch_size),
            finetune_flow=bool(args.finetune_flow),
        )
        train_log.append(
            {
                "update": it,
                "mean_reward": float(buf["rewards"].mean()),
                **st,
            }
        )
        print(
            f"[reinflow_rl] update {it+1}/{args.total_updates} "
            f"mean_r={buf['rewards'].mean():.4f} pg={st['pg']:.4f} v={st['v']:.4f}"
        )

    ckpt_out = out_dir / "reinflow_rl_checkpoint.pt"
    torch.save(
        {
            "log_std_head": log_std_head.state_dict(),
            "value_net": value_net.state_dict(),
            "finetune_flow": bool(args.finetune_flow),
            "bc_checkpoint": str(pathlib.Path(args.bc_checkpoint).resolve()),
            "train_log": train_log,
        },
        ckpt_out,
    )
    torch.save(flow.state_dict(), out_dir / "flow_policy_after_rl.pt")

    with open(out_dir / "training_final.json", "w") as f:
        json.dump(
            {
                "train_loss_final": float(train_log[-1]["pg"]) if train_log else 0.0,
                "val_loss_final": float(train_log[-1]["v"]) if train_log else 0.0,
                "kind": "reinflow_rl_ppo",
            },
            f,
        )
    with open(out_dir / "reinflow_rl_train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    metrics_path = out_dir / "metrics.json"
    n_max = max(int(args.n_train_val_episodes), int(args.n_infer_episodes))
    runner_eval = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=str(out_dir),
        eval_episodes=n_max,
    )
    try:
        pol_eval.flow.eval()
        m_te = runner_eval.run_eval_metrics(
            pol_eval,
            warmup_predict_steps=20,
            eval_seed=int(args.seed),
            log_video=False,
            n_episodes=int(args.n_infer_episodes),
            save_inference_videos_dir=None
            if args.skip_inference_videos
            else str((out_dir / "inference_videos").resolve()),
        )
        if int(args.n_train_val_episodes) > 0:
            m_tv = runner_eval.run_eval_metrics(
                pol_eval,
                warmup_predict_steps=20,
                eval_seed=int(args.seed + args.train_val_eval_seed_offset),
                log_video=False,
                n_episodes=int(args.n_train_val_episodes),
                save_inference_videos_dir=None,
            )
            from infer_kitchen import _merge_phases  # noqa: E402

            serializable = {**_merge_phases(m_tv, m_te)}
        else:
            from infer_kitchen import _legacy_from_test, _prefix_metrics  # noqa: E402

            serializable = {
                **_prefix_metrics("test", m_te),
                **_legacy_from_test(m_te),
            }
        with open(metrics_path, "w") as f:
            json.dump(serializable, f, indent=2)
    finally:
        try:
            runner.close()
        except Exception:
            pass
        try:
            runner_eval.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
