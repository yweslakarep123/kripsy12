#!/usr/bin/env python3
"""
Orkestrator eksperimen:
  1) Baseline FlowPolicy (hyperparameter default) × seed × preprocessing × CV — selalu lebih dulu
  2) Random search configs × seed × preprocessing × CV

Resume: metrik lengkap (metrics.json atau baris results.csv status=ok) dilewati;
training terputus dilanjutkan (resume Hydra) jika ada latest.ckpt tanpa training_final.json;
infer saja jika training sudah selesai (training_final.json + ckpt) tanpa metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FLOWPOLICY_ROOT = REPO_ROOT / "FlowPolicy"

sys.path.insert(0, str(SCRIPT_DIR))
from cv_splits import build_cv_splits, save_splits  # noqa: E402
from experiment_constants import (  # noqa: E402
    BASELINE_CFG_IDX,
    CSV_HPARAM_KEYS,
    SEARCH_SPACE,
    baseline_config_dict,
    compute_horizon,
    sample_configs,
)


def _fmt_hydra_val(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return repr(float(v))
    return str(v)


def apply_vram_limits(cfg: Dict[str, Any], max_batch: int) -> Dict[str, Any]:
    c = dict(cfg)
    c["dataloader.batch_size"] = min(int(c["dataloader.batch_size"]), int(max_batch))
    return c


def load_or_create_config_bundle(
    configs_path: pathlib.Path,
    sampling_seed: int,
    n_configs: int,
    max_batch: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (baseline_cfg, sampled_cfgs). Migrasi dari JSON array lama → objek {baseline, sampled}."""
    baseline = apply_vram_limits(baseline_config_dict(), max_batch)

    if configs_path.is_file():
        with open(configs_path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            sampled = [apply_vram_limits(dict(x), max_batch) for x in raw]
            bundle = {"version": 2, "baseline": baseline, "sampled": sampled}
            with open(configs_path, "w") as fw:
                json.dump(bundle, fw, indent=2)
            print(
                "[info] configs.json format lama (array) dimigrasi ke {baseline, sampled}."
            )
            return baseline, sampled
        if isinstance(raw, dict) and "sampled" in raw:
            b = raw.get("baseline")
            if isinstance(b, dict):
                baseline = apply_vram_limits({**baseline, **b, "cfg_idx": BASELINE_CFG_IDX}, max_batch)
            sampled = [
                apply_vram_limits(dict(x), max_batch) for x in raw["sampled"]
            ]
            return baseline, sampled

    rng = np.random.RandomState(sampling_seed)
    sampled = [apply_vram_limits(c, max_batch) for c in sample_configs(rng, n_configs)]
    bundle = {"version": 2, "baseline": baseline, "sampled": sampled}
    configs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(configs_path, "w") as f:
        json.dump(bundle, f, indent=2)
    return baseline, sampled


def build_train_overrides(
    cfg: Dict[str, Any],
    *,
    seed: int,
    profile: str,
    train_eps: List[int],
    val_eps: List[int],
    run_dir: pathlib.Path,
    zarr_rel: str,
    resume_training: bool,
    checkpoint_every: int,
    dataloader_num_workers: int,
) -> List[str]:
    n_obs = int(cfg["n_obs_steps"])
    n_act = int(cfg["n_action_steps"])
    hz = compute_horizon(n_obs, n_act)
    bs = int(cfg["dataloader.batch_size"])

    def il(xs: List[int]) -> str:
        return "[" + ",".join(str(int(x)) for x in xs) + "]"

    odl: List[str] = [
        "task=franka_kitchen_complete4",
        f"task.dataset.zarr_path={zarr_rel}",
        f"task.dataset.train_episode_indices={il(train_eps)}",
        f"task.dataset.val_episode_indices={il(val_eps)}",
        f"task.dataset.preprocessing_profile={profile}",
        f"training.seed={seed}",
        f"task.dataset.seed={seed}",
        "training.compute_val_loss=true",
        "training.rollout_every=999999",
        f"training.resume={str(resume_training).lower()}",
        "checkpoint.save_ckpt=true",
        f"training.checkpoint_every={checkpoint_every}",
        "checkpoint.save_last_ckpt=true",
        "logging.mode=offline",
        f"hydra.run.dir={run_dir.resolve()}",
        "hydra.job.chdir=true",
        f"horizon={hz}",
        f"n_obs_steps={n_obs}",
        f"n_action_steps={n_act}",
        f"dataloader.batch_size={bs}",
        f"val_dataloader.batch_size={bs}",
        f"dataloader.num_workers={dataloader_num_workers}",
        f"val_dataloader.num_workers={dataloader_num_workers}",
    ]

    for k in CSV_HPARAM_KEYS:
        if k == "cfg_idx":
            continue
        if k == "_state_mlp_hidden":
            odl.append(f"policy.encoder_output_dim={_fmt_hydra_val(cfg[k])}")
            continue
        odl.append(f"{k}={_fmt_hydra_val(cfg[k])}")
    return odl


def row_key_ok_exists(csv_path: pathlib.Path, key: Tuple[int, int, str, int]) -> bool:
    if not csv_path.is_file():
        return False
    df = pd.read_csv(csv_path)
    if df.empty:
        return False
    df["cfg_idx"] = df["cfg_idx"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["fold"] = df["fold"].astype(int)
    m = (
        (df["cfg_idx"] == int(key[0]))
        & (df["seed"] == int(key[1]))
        & (df["profile"].astype(str) == str(key[2]))
        & (df["fold"] == int(key[3]))
        & (df["status"].astype(str) == "ok")
    )
    return bool(m.any())


def append_results_csv(
    csv_path: pathlib.Path,
    row: Dict[str, Any],
    hp_cols: List[str],
) -> None:
    fieldnames = (
        ["cfg_idx", "seed", "profile", "fold"]
        + hp_cols
        + [
            "success_rate_k1",
            "success_rate_k2",
            "success_rate_k3",
            "success_rate_k4",
            "mean_inference_latency_ms",
            "std_inference_latency_ms",
            "trade_off",
            "train_loss_final",
            "val_loss_final",
            "n_infer_episodes",
            "checkpoint_path",
            "status",
        ]
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_training_final(run_dir: pathlib.Path) -> Tuple[Any, Any]:
    p = run_dir / "training_final.json"
    if not p.is_file():
        return None, None
    with open(p) as f:
        d = json.load(f)
    return d.get("train_loss_final"), d.get("val_loss_final")


def print_run_configuration(
    label: str,
    cfg: Dict[str, Any],
    overrides: List[str],
    run_dir: pathlib.Path,
) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("Folder run:", run_dir.resolve())
    print("-" * 72)
    print("Hyperparameter (flat):")
    hp = {k: cfg[k] for k in CSV_HPARAM_KEYS if k in cfg}
    print(json.dumps(hp, indent=2, default=str))
    print("-" * 72)
    print("Override Hydra (train):")
    for line in sorted(overrides):
        print(" ", line)
    print("=" * 72 + "\n")


def sync_csv_from_metrics_if_needed(
    results_csv: pathlib.Path,
    hp_cols: List[str],
    cfg: Dict[str, Any],
    cfg_idx: int,
    seed: int,
    profile: str,
    fold_i: int,
    run_dir: pathlib.Path,
    ckpt_path: pathlib.Path,
    metrics_path: pathlib.Path,
) -> None:
    rk = (cfg_idx, seed, profile, fold_i)
    if row_key_ok_exists(results_csv, rk):
        return
    with open(metrics_path) as f:
        met = json.load(f)
    tr_l, va_l = load_training_final(run_dir)
    append_results_csv(
        results_csv,
        {
            "cfg_idx": cfg_idx,
            "seed": seed,
            "profile": profile,
            "fold": fold_i,
            **{k: cfg[k] for k in hp_cols},
            "success_rate_k1": met.get("success_rate_k1"),
            "success_rate_k2": met.get("success_rate_k2"),
            "success_rate_k3": met.get("success_rate_k3"),
            "success_rate_k4": met.get("success_rate_k4"),
            "mean_inference_latency_ms": met.get("mean_inference_latency_ms"),
            "std_inference_latency_ms": met.get("std_inference_latency_ms"),
            "trade_off": met.get("trade_off"),
            "train_loss_final": tr_l,
            "val_loss_final": va_l,
            "n_infer_episodes": met.get("n_infer_episodes"),
            "checkpoint_path": str(ckpt_path),
            "status": "skipped_resume",
        },
        hp_cols,
    )


def run_infer_subprocess(
    py: str,
    infer_py: pathlib.Path,
    cwd_train: str,
    env: dict,
    ckpt_path: pathlib.Path,
    metrics_path: pathlib.Path,
    n_infer_episodes: int,
    seed: int,
) -> int:
    return subprocess.run(
        [
            py,
            str(infer_py),
            "--checkpoint",
            str(ckpt_path),
            "--metrics-json",
            str(metrics_path),
            "--n-infer-episodes",
            str(n_infer_episodes),
            "--seed",
            str(seed),
            "--warmup-steps",
            "20",
        ],
        cwd=cwd_train,
        env=env,
    ).returncode


def execute_one_job(
    *,
    cfg: Dict[str, Any],
    cfg_idx: int,
    seed: int,
    profile: str,
    fold_i: int,
    fold_entry: Dict[str, Any],
    run_name: str,
    runs_root: pathlib.Path,
    results_csv: pathlib.Path,
    hp_cols: List[str],
    py: str,
    train_py: pathlib.Path,
    infer_py: pathlib.Path,
    cwd_train: str,
    zarr_path: str,
    n_infer_episodes: int,
    checkpoint_every: int,
    dataloader_num_workers: int,
) -> None:
    run_dir = runs_root / run_name
    metrics_path = run_dir / "metrics.json"
    ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
    training_final_path = run_dir / "training_final.json"
    rk = (cfg_idx, seed, profile, fold_i)

    if metrics_path.is_file():
        print(f"[skip] {run_name}: infer selesai (metrics.json ada)")
        sync_csv_from_metrics_if_needed(
            results_csv,
            hp_cols,
            cfg,
            cfg_idx,
            seed,
            profile,
            fold_i,
            run_dir,
            ckpt_path,
            metrics_path,
        )
        return

    if row_key_ok_exists(results_csv, rk):
        print(f"[skip] {run_name}: sudah tercatat status=ok di results.csv")
        return

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")

    infer_only = (
        ckpt_path.is_file()
        and training_final_path.is_file()
        and not metrics_path.is_file()
    )

    if infer_only:
        print(f"[infer-only] {run_name}: training_final.json + ckpt ada, lanjut inferensi")
        rc = run_infer_subprocess(
            py,
            infer_py,
            cwd_train,
            env,
            ckpt_path,
            metrics_path,
            n_infer_episodes,
            seed,
        )
        tr_l, va_l = load_training_final(run_dir)
        if rc != 0 or not metrics_path.is_file():
            append_results_csv(
                results_csv,
                {
                    "cfg_idx": cfg_idx,
                    "seed": seed,
                    "profile": profile,
                    "fold": fold_i,
                    **{k: cfg[k] for k in hp_cols},
                    "success_rate_k1": "",
                    "success_rate_k2": "",
                    "success_rate_k3": "",
                    "success_rate_k4": "",
                    "mean_inference_latency_ms": "",
                    "std_inference_latency_ms": "",
                    "trade_off": "",
                    "train_loss_final": tr_l,
                    "val_loss_final": va_l,
                    "n_infer_episodes": n_infer_episodes,
                    "checkpoint_path": str(ckpt_path),
                    "status": f"infer_failed_{rc}",
                },
                hp_cols,
            )
            return
        with open(metrics_path) as f:
            met = json.load(f)
        append_results_csv(
            results_csv,
            {
                "cfg_idx": cfg_idx,
                "seed": seed,
                "profile": profile,
                "fold": fold_i,
                **{k: cfg[k] for k in hp_cols},
                "success_rate_k1": met.get("success_rate_k1"),
                "success_rate_k2": met.get("success_rate_k2"),
                "success_rate_k3": met.get("success_rate_k3"),
                "success_rate_k4": met.get("success_rate_k4"),
                "mean_inference_latency_ms": met.get("mean_inference_latency_ms"),
                "std_inference_latency_ms": met.get("std_inference_latency_ms"),
                "trade_off": met.get("trade_off"),
                "train_loss_final": tr_l,
                "val_loss_final": va_l,
                "n_infer_episodes": met.get("n_infer_episodes", n_infer_episodes),
                "checkpoint_path": str(ckpt_path),
                "status": "ok",
            },
            hp_cols,
        )
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    resume_training = bool(
        ckpt_path.is_file() and not training_final_path.is_file()
    )
    if resume_training:
        print(f"[resume] {run_name}: melanjutkan training dari checkpoints/latest.ckpt")

    overrides = build_train_overrides(
        cfg,
        seed=seed,
        profile=profile,
        train_eps=fold_entry["train_episodes"],
        val_eps=fold_entry["val_episodes"],
        run_dir=run_dir,
        zarr_rel=zarr_path,
        resume_training=resume_training,
        checkpoint_every=checkpoint_every,
        dataloader_num_workers=dataloader_num_workers,
    )

    phase = "BASELINE (default)" if cfg_idx == BASELINE_CFG_IDX else f"Random search cfg_idx={cfg_idx}"
    print_run_configuration(
        f"[train] {run_name}  |  {phase}",
        cfg,
        overrides,
        run_dir,
    )

    r = subprocess.run([py, str(train_py)] + overrides, cwd=cwd_train, env=env)
    if r.returncode != 0:
        append_results_csv(
            results_csv,
            {
                "cfg_idx": cfg_idx,
                "seed": seed,
                "profile": profile,
                "fold": fold_i,
                **{k: cfg[k] for k in hp_cols},
                "success_rate_k1": "",
                "success_rate_k2": "",
                "success_rate_k3": "",
                "success_rate_k4": "",
                "mean_inference_latency_ms": "",
                "std_inference_latency_ms": "",
                "trade_off": "",
                "train_loss_final": "",
                "val_loss_final": "",
                "n_infer_episodes": "",
                "checkpoint_path": str(ckpt_path),
                "status": f"train_failed_{r.returncode}",
            },
            hp_cols,
        )
        return

    if not ckpt_path.is_file():
        append_results_csv(
            results_csv,
            {
                "cfg_idx": cfg_idx,
                "seed": seed,
                "profile": profile,
                "fold": fold_i,
                **{k: cfg[k] for k in hp_cols},
                "success_rate_k1": "",
                "success_rate_k2": "",
                "success_rate_k3": "",
                "success_rate_k4": "",
                "mean_inference_latency_ms": "",
                "std_inference_latency_ms": "",
                "trade_off": "",
                "train_loss_final": "",
                "val_loss_final": "",
                "n_infer_episodes": "",
                "checkpoint_path": str(ckpt_path),
                "status": "no_checkpoint",
            },
            hp_cols,
        )
        return

    print_run_configuration(
        f"[infer] {run_name}",
        cfg,
        [
            f"checkpoint={ckpt_path}",
            f"metrics_json={metrics_path}",
            f"n_infer_episodes={n_infer_episodes}",
            f"seed={seed}",
        ],
        run_dir,
    )
    r2 = run_infer_subprocess(
        py,
        infer_py,
        cwd_train,
        env,
        ckpt_path,
        metrics_path,
        n_infer_episodes,
        seed,
    )
    tr_l, va_l = load_training_final(run_dir)
    if r2 != 0 or not metrics_path.is_file():
        append_results_csv(
            results_csv,
            {
                "cfg_idx": cfg_idx,
                "seed": seed,
                "profile": profile,
                "fold": fold_i,
                **{k: cfg[k] for k in hp_cols},
                "success_rate_k1": "",
                "success_rate_k2": "",
                "success_rate_k3": "",
                "success_rate_k4": "",
                "mean_inference_latency_ms": "",
                "std_inference_latency_ms": "",
                "trade_off": "",
                "train_loss_final": tr_l,
                "val_loss_final": va_l,
                "n_infer_episodes": n_infer_episodes,
                "checkpoint_path": str(ckpt_path),
                "status": f"infer_failed_{r2}",
            },
            hp_cols,
        )
        return

    with open(metrics_path) as f:
        met = json.load(f)

    append_results_csv(
        results_csv,
        {
            "cfg_idx": cfg_idx,
            "seed": seed,
            "profile": profile,
            "fold": fold_i,
            **{k: cfg[k] for k in hp_cols},
            "success_rate_k1": met.get("success_rate_k1"),
            "success_rate_k2": met.get("success_rate_k2"),
            "success_rate_k3": met.get("success_rate_k3"),
            "success_rate_k4": met.get("success_rate_k4"),
            "mean_inference_latency_ms": met.get("mean_inference_latency_ms"),
            "std_inference_latency_ms": met.get("std_inference_latency_ms"),
            "trade_off": met.get("trade_off"),
            "train_loss_final": tr_l,
            "val_loss_final": va_l,
            "n_infer_episodes": met.get("n_infer_episodes", n_infer_episodes),
            "checkpoint_path": str(ckpt_path),
            "status": "ok",
        },
        hp_cols,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 101])
    ap.add_argument(
        "--profiles", type=str, nargs="+", default=["standard", "minimal"]
    )
    ap.add_argument("--n-configs", type=int, default=10)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--sampling-seed", type=int, default=99)
    ap.add_argument("--cv-seed", type=int, default=12345)
    ap.add_argument("--n-infer-episodes", type=int, default=50)
    ap.add_argument("--output-dir", type=str, default="outputs/experiment")
    ap.add_argument(
        "--zarr-path",
        type=str,
        default="data/kitchen_complete_from_minari.zarr",
        help="Relatif ke direktori FlowPolicy/FlowPolicy (tempat train.py)",
    )
    ap.add_argument("--n-episodes", type=int, default=19)
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Plafon batch size (training+val) untuk mengurangi risiko OOM pada VRAM ~16GB.",
    )
    ap.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Kurangi memori CPU/host; turunkan jika RAM habis.",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Simpan checkpoint berkala agar training bisa dilanjut setelah mesin mati.",
    )
    args = ap.parse_args()

    out_root = (REPO_ROOT / args.output_dir).resolve()
    runs_root = out_root / "runs"
    plots_dir = out_root / "plots"
    runs_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs_path = out_root / "configs.json"
    cv_path = out_root / "cv_splits.json"
    results_csv = out_root / "results.csv"

    baseline_cfg, sampled_cfgs = load_or_create_config_bundle(
        configs_path,
        args.sampling_seed,
        args.n_configs,
        args.max_batch_size,
    )

    folds = build_cv_splits(
        n_episodes=args.n_episodes,
        n_folds=args.n_folds,
        held_out_test=1,
        seed=args.cv_seed,
    )
    save_splits(
        str(cv_path),
        folds,
        meta={
            "n_episodes": args.n_episodes,
            "n_folds": args.n_folds,
            "cv_seed": args.cv_seed,
            "sampling_seed": args.sampling_seed,
            "max_batch_size": args.max_batch_size,
        },
    )

    py = sys.executable
    train_py = FLOWPOLICY_ROOT / "train.py"
    infer_py = FLOWPOLICY_ROOT / "infer_kitchen.py"
    cwd_train = str(FLOWPOLICY_ROOT.resolve())
    hp_cols = list(SEARCH_SPACE.keys())

    all_cfgs = [baseline_cfg] + sampled_cfgs

    print(
        "\n>>> Urutan eksperimen: (1) BASELINE default Hyperparameter → "
        "(2) random search\n"
        f"    VRAM safety: max_batch_size={args.max_batch_size}, "
        f"num_workers={args.dataloader_num_workers}\n"
    )

    for cfg in all_cfgs:
        cfg_idx = int(cfg["cfg_idx"])
        for seed in args.seeds:
            for profile in args.profiles:
                for fold_entry in folds:
                    fold_i = int(fold_entry["fold"])
                    if cfg_idx == BASELINE_CFG_IDX:
                        run_name = f"baseline_seed{seed}_{profile}_fold{fold_i}"
                    else:
                        run_name = f"cfg{cfg_idx}_seed{seed}_{profile}_fold{fold_i}"

                    execute_one_job(
                        cfg=cfg,
                        cfg_idx=cfg_idx,
                        seed=seed,
                        profile=profile,
                        fold_i=fold_i,
                        fold_entry=fold_entry,
                        run_name=run_name,
                        runs_root=runs_root,
                        results_csv=results_csv,
                        hp_cols=hp_cols,
                        py=py,
                        train_py=train_py,
                        infer_py=infer_py,
                        cwd_train=cwd_train,
                        zarr_path=args.zarr_path,
                        n_infer_episodes=args.n_infer_episodes,
                        checkpoint_every=args.checkpoint_every,
                        dataloader_num_workers=args.dataloader_num_workers,
                    )

    summarize_script = SCRIPT_DIR / "summarize.py"
    plot_script = SCRIPT_DIR / "plot_results.py"
    subprocess.run(
        [py, str(summarize_script), "--output-dir", str(out_root)], check=False
    )
    subprocess.run([py, str(plot_script), "--output-dir", str(out_root)], check=False)


if __name__ == "__main__":
    main()
