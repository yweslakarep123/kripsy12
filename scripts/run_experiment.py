#!/usr/bin/env python3
"""
Orkestrator eksperimen (tanpa k-fold, satu partisi train/val/test):

  1) Baseline — hyperparameter default × len(seeds) × len(profiles)
  2) Pencarian — **random search berpusat di baseline**.

Random search berjalan pada **satu seed × satu profile** (default: seed=0,
profile=standard — sama dengan satu run baseline) menggunakan ``val_loss`` untuk memilih pemenang.
Setelah selesai, konfigurasi pemenang di-**rerun penuh**
pada semua ``seeds × profiles`` user (default 3 × 2 = 6 run) dengan training
+ inference + write ``results.csv`` ``status=ok`` — analog baseline.

Flag mutually exclusive:

- ``--baseline-only`` — hanya baseline; random search dilewati.
- ``--search-only`` — hanya random search + rerun pemenang (skip baseline).

Tanpa flag: jalankan baseline lalu random search berurutan.

Metrik inferensi: fase train/val (sim) + fase test; metrik simulasi akhir
training (``training_sim_*``) dari ``training_sim_metrics.json``; success total
& k1–k4; latensi global + rata-rata per-episod; ``trade_off`` dan
``trade_off_episode_latency``. Video: MP4 inferensi per-episod di
``inference_videos/``, bukan video rollout training di W&B.

Resume:

- Baseline & pemenang search (cfg_idx=-3): metrik lengkap (``metrics.json``)
  dilewati; juga dilewati jika baris ``results.csv`` yang sama sudah ``status=ok``.
- Training terputus dilanjutkan (resume Hydra) jika ada ``latest.ckpt`` tanpa
  ``training_final.json``; infer saja jika ``training_final.json`` + ckpt sudah
  ada tetapi belum ``metrics.json``.
- Random search: ``random_search_state.json`` di ``--output-dir`` — resume otomatis.
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

import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FLOWPOLICY_ROOT = REPO_ROOT / "FlowPolicy"

sys.path.insert(0, str(SCRIPT_DIR))
from cv_splits import build_single_train_val_split, save_splits  # noqa: E402
from experiment_constants import (  # noqa: E402
    BASELINE_CFG_IDX,
    CSV_HPARAM_KEYS,
    RESULTS_CSV_METRIC_COLUMNS,
    SEARCH_BEST_CFG_IDX,
    baseline_config_dict,
    empty_metrics_row,
    metrics_row_from_infer_json,
)
from random_search import run_random_search  # noqa: E402
from train_overrides import build_train_overrides  # noqa: E402


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
    max_batch: int,
) -> Dict[str, Any]:
    """Muat / buat ``configs.json`` (``version: 5``) dengan baseline saja.

    Random search menyimpan state di ``random_search_state.json``.
    File ini hanya menyimpan baseline untuk fase-1 dan re-run pemenang.
    """
    baseline = apply_vram_limits(baseline_config_dict(), max_batch)

    raw: Any = None
    if configs_path.is_file():
        text = configs_path.read_text(encoding="utf-8").strip()
        if not text:
            print("[warn] configs.json kosong; akan dibuat ulang.")
        elif text:
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as e:
                print(f"[warn] configs.json bukan JSON valid ({e}); akan dibuat ulang.")

    if isinstance(raw, dict) and isinstance(raw.get("baseline"), dict):
        b = raw["baseline"]
        baseline = apply_vram_limits(
            {**baseline, **b, "cfg_idx": BASELINE_CFG_IDX}, max_batch
        )

    configs_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": 5,
        "search_mode": "random_search_around_baseline",
        "baseline": baseline,
    }
    with open(configs_path, "w") as f:
        json.dump(bundle, f, indent=2)
    return baseline


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
        + list(RESULTS_CSV_METRIC_COLUMNS)
        + [
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
    mrow = metrics_row_from_infer_json(met)
    append_results_csv(
        results_csv,
        {
            "cfg_idx": cfg_idx,
            "seed": seed,
            "profile": profile,
            "fold": fold_i,
            **{k: cfg[k] for k in hp_cols},
            **mrow,
            "train_loss_final": tr_l,
            "val_loss_final": va_l,
            "n_infer_episodes": met.get(
                "test_n_infer_episodes",
                met.get("n_infer_episodes"),
            ),
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
    *,
    n_train_val_episodes: int,
    train_val_eval_seed_offset: int,
    skip_inference_videos: bool = False,
) -> int:
    cmd = [
        py,
        str(infer_py),
        "--checkpoint",
        str(ckpt_path),
        "--metrics-json",
        str(metrics_path),
        "--n-train-val-episodes",
        str(int(n_train_val_episodes)),
        "--train-val-eval-seed-offset",
        str(int(train_val_eval_seed_offset)),
        "--n-infer-episodes",
        str(n_infer_episodes),
        "--seed",
        str(seed),
        "--warmup-steps",
        "20",
    ]
    if skip_inference_videos:
        cmd.append("--skip-inference-videos")
    else:
        vdir = pathlib.Path(metrics_path).parent / "inference_videos"
        cmd.extend(["--inference-videos-dir", str(vdir.resolve())])
    return subprocess.run(cmd, cwd=cwd_train, env=env).returncode


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
    n_train_val_episodes: int,
    train_val_eval_seed_offset: int,
    skip_inference_videos: bool = False,
    resume_from_results_csv: bool = True,
    enable_early_stop: bool = True,
    early_stop_rollout_every: int = 200,
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

    if resume_from_results_csv and row_key_ok_exists(results_csv, rk):
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
            n_train_val_episodes=n_train_val_episodes,
            train_val_eval_seed_offset=train_val_eval_seed_offset,
            skip_inference_videos=skip_inference_videos,
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
                    **empty_metrics_row(),
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
        mrow = metrics_row_from_infer_json(met)
        append_results_csv(
            results_csv,
            {
                "cfg_idx": cfg_idx,
                "seed": seed,
                "profile": profile,
                "fold": fold_i,
                **{k: cfg[k] for k in hp_cols},
                **mrow,
                "train_loss_final": tr_l,
                "val_loss_final": va_l,
                "n_infer_episodes": met.get(
                    "test_n_infer_episodes",
                    met.get("n_infer_episodes", n_infer_episodes),
                ),
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
        enable_early_stop=enable_early_stop,
        early_stop_rollout_every=early_stop_rollout_every,
    )

    phase = (
        "BASELINE (default)"
        if cfg_idx == BASELINE_CFG_IDX
        else f"Pencarian hiperparameter cfg_idx={cfg_idx}"
    )
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
                **empty_metrics_row(),
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
                **empty_metrics_row(),
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
        n_train_val_episodes=n_train_val_episodes,
        train_val_eval_seed_offset=train_val_eval_seed_offset,
        skip_inference_videos=skip_inference_videos,
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
                **empty_metrics_row(),
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
    mrow = metrics_row_from_infer_json(met)
    append_results_csv(
        results_csv,
        {
            "cfg_idx": cfg_idx,
            "seed": seed,
            "profile": profile,
            "fold": fold_i,
            **{k: cfg[k] for k in hp_cols},
            **mrow,
            "train_loss_final": tr_l,
            "val_loss_final": va_l,
            "n_infer_episodes": met.get(
                "test_n_infer_episodes",
                met.get("n_infer_episodes", n_infer_episodes),
            ),
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
    ap.add_argument(
        "--cv-seed",
        type=int,
        default=12345,
        help="Seed pembagian episode train/val (satu partisi, tanpa k-fold).",
    )
    ap.add_argument("--n-infer-episodes", type=int, default=50)
    ap.add_argument("--output-dir", type=str, default="outputs/experiment")
    ap.add_argument(
        "--results-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Jalur results.csv (relatif ke akar repo atau absolut). "
        "Default: <output-dir>/results.csv. Jika diisi, semua fase menulis ke file "
        "ini dan melewati job (cfg_idx, seed, profile, fold) yang sudah status=ok.",
    )
    ap.add_argument(
        "--zarr-path",
        type=str,
        default="FlowPolicy/data/kitchen_complete_from_minari.zarr",
        help="Relatif ke akar paket (folder berisi train.py dan flow_policy_3d/), "
        "mis. FlowPolicy/data/... → .../FlowPolicy/FlowPolicy/data/...",
    )
    ap.add_argument("--n-episodes", type=int, default=19)
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Plafon batch size baseline + rerun pemenang (default 128; VRAM ~16GB).",
    )
    ap.add_argument(
        "--search-max-batch-size",
        type=int,
        default=512,
        help="Plafon batch size khusus fase random search (default 512).",
    )
    ap.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Kurangi memori CPU/host; turunkan jika RAM habis.",
    )
    ap.add_argument(
        "--baseline-only",
        action="store_true",
        help="Hanya baseline (3 seed × 2 profil = 6 run default); tanpa random search.",
    )
    ap.add_argument(
        "--search-only",
        action="store_true",
        help="Hanya random search + re-run pemenang top-1 (tanpa baseline).",
    )
    ap.add_argument(
        "--hyperband-only",
        action="store_true",
        help="Alias untuk --search-only (kompatibilitas lama).",
    )
    ap.add_argument(
        "--random-search-n",
        type=int,
        default=16,
        metavar="N",
        help="Jumlah trial random search berpusat di baseline (default 16).",
    )
    ap.add_argument(
        "--random-search-seed",
        type=int,
        default=99,
        help="Seed RNG sampling konfigurasi random search.",
    )
    ap.add_argument(
        "--random-search-sigma",
        type=float,
        default=1.0,
        help="Lebar Gaussian sampling di sekitar baseline (indeks ruang diskret).",
    )
    ap.add_argument(
        "--random-search-p-exact-baseline",
        type=float,
        default=0.15,
        help="Probabilitas trial persis baseline (default 0.15).",
    )
    ap.add_argument(
        "--search-train-seed",
        type=int,
        default=0,
        help="Seed training SELAMA fase random search (1 seed agar cepat).",
    )
    ap.add_argument(
        "--search-profile",
        type=str,
        default="minimal",
        help="Profil preprocessing SELAMA fase random search (1 profil saja; default: minimal).",
    )
    ap.add_argument(
        "--disable-early-stop",
        action="store_true",
        help="Nonaktifkan early stopping success-rate (baseline, random search, dan rerun).",
    )
    ap.add_argument(
        "--early-stop-rollout-every",
        type=int,
        default=200,
        help="Interval epoch eval simulasi untuk early stopping (default 200).",
    )
    ap.add_argument(
        "--n-train-val-episodes",
        type=int,
        default=15,
        help="Episode simulasi untuk metrik fase train/val (infer_kitchen); 0 = lewati.",
    )
    ap.add_argument(
        "--train-val-eval-seed-offset",
        type=int,
        default=31,
        help="Offset seed eval train/val vs test (infer_kitchen).",
    )
    ap.add_argument(
        "--skip-inference-videos",
        action="store_true",
        help="Jangan simpan MP4 infer_ep_*.mp4 (hemat waktu/ruang).",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Simpan checkpoint berkala agar training bisa dilanjut setelah mesin mati.",
    )
    args = ap.parse_args()
    search_only = bool(args.search_only or args.hyperband_only)
    if args.baseline_only and search_only:
        ap.error("--baseline-only dan --search-only/--hyperband-only saling meniadakan.")

    out_root = (REPO_ROOT / args.output_dir).resolve()
    runs_root = out_root / "runs"
    plots_dir = out_root / "plots"
    runs_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs_path = out_root / "configs.json"
    cv_path = out_root / "cv_splits.json"
    if args.results_csv:
        _rcp = pathlib.Path(args.results_csv)
        results_csv = (
            _rcp.resolve() if _rcp.is_absolute() else (REPO_ROOT / _rcp).resolve()
        )
        results_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_csv = out_root / "results.csv"

    baseline_cfg = load_or_create_config_bundle(configs_path, args.max_batch_size)

    fold_entry = build_single_train_val_split(
        n_episodes=args.n_episodes,
        held_out_test=1,
        n_grid_partitions=5,
        partition_index=0,
        seed=args.cv_seed,
    )
    save_splits(
        str(cv_path),
        [fold_entry],
        meta={
            "n_episodes": args.n_episodes,
            "split_mode": "single_train_val",
            "n_grid_partitions": 5,
            "partition_index": 0,
            "cv_seed": args.cv_seed,
            "max_batch_size": args.max_batch_size,
            "hyperparam_search": "random_search_around_baseline",
            "random_search_n": int(args.random_search_n),
            "random_search_seed": int(args.random_search_seed),
            "random_search_sigma": float(args.random_search_sigma),
        },
    )

    py = sys.executable
    train_py = FLOWPOLICY_ROOT / "train.py"
    infer_py = FLOWPOLICY_ROOT / "infer_kitchen.py"
    cwd_train = str(FLOWPOLICY_ROOT.resolve())
    hp_cols = list(CSV_HPARAM_KEYS)
    split_fold_idx = int(fold_entry["fold"])

    enable_early_stop = not args.disable_early_stop

    n_base = len(args.seeds) * len(args.profiles)
    n_final = len(args.seeds) * len(args.profiles)
    if args.baseline_only:
        print(
            "\n>>> Mode --baseline-only: hanya baseline "
            f"({n_base} run). Random search dilewati.\n"
            "    Satu partisi train/val, tanpa k-fold.\n"
            f"    Early stop: {'on' if enable_early_stop else 'off'}, "
            f"rollout_every={args.early_stop_rollout_every}\n"
            f"    VRAM baseline/rerun: max_batch_size={args.max_batch_size}, "
            f"random search: search_max_batch_size={args.search_max_batch_size}, "
            f"num_workers={args.dataloader_num_workers}\n"
        )
    elif search_only:
        print(
            "\n>>> Mode --search-only: random search (1 seed × 1 profile) "
            f"diikuti rerun top-1 pemenang di {n_final} run "
            f"({len(args.seeds)} seeds × {len(args.profiles)} profiles).\n"
            "    Baseline dilewati.\n"
            f"    N={args.random_search_n}, seed={args.random_search_seed}, "
            f"sigma={args.random_search_sigma}\n"
            f"    Early stop random search: {'on' if enable_early_stop else 'off'}, "
            f"rollout_every={args.early_stop_rollout_every}\n"
            f"    VRAM baseline/rerun: max_batch_size={args.max_batch_size}, "
            f"random search: search_max_batch_size={args.search_max_batch_size}, "
            f"num_workers={args.dataloader_num_workers}\n"
        )
    else:
        print(
            "\n>>> Urutan: (1) Baseline "
            f"({n_base} run) → (2) random search (1 seed × 1 profile) "
            f"→ (3) rerun top-1 pemenang ({n_final} run). "
            "Satu partisi train/val, tanpa k-fold.\n"
            f"    Random search: N={args.random_search_n}, "
            f"seed={args.random_search_seed}, sigma={args.random_search_sigma}\n"
            f"    Early stop: {'on' if enable_early_stop else 'off'}, "
            f"rollout_every={args.early_stop_rollout_every}\n"
            f"    VRAM baseline/rerun: max_batch_size={args.max_batch_size}, "
            f"random search: search_max_batch_size={args.search_max_batch_size}, "
            f"num_workers={args.dataloader_num_workers}\n"
        )

    def run_grid_for_configs(
        cfgs: List[Dict[str, Any]],
    ) -> None:
        for cfg in cfgs:
            cfg_idx = int(cfg["cfg_idx"])
            for seed in args.seeds:
                for profile in args.profiles:
                    if cfg_idx == BASELINE_CFG_IDX:
                        run_name = f"baseline_seed{seed}_{profile}"
                    elif cfg_idx == SEARCH_BEST_CFG_IDX:
                        run_name = f"search_best_seed{seed}_{profile}"
                    else:
                        run_name = f"cfg{cfg_idx}_seed{seed}_{profile}"
                    execute_one_job(
                        cfg=cfg,
                        cfg_idx=cfg_idx,
                        seed=seed,
                        profile=profile,
                        fold_i=split_fold_idx,
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
                        n_train_val_episodes=args.n_train_val_episodes,
                        train_val_eval_seed_offset=args.train_val_eval_seed_offset,
                        skip_inference_videos=args.skip_inference_videos,
                        resume_from_results_csv=True,
                        enable_early_stop=enable_early_stop,
                        early_stop_rollout_every=args.early_stop_rollout_every,
                    )

    def run_random_search_phase() -> None:
        """Jalankan random search (single seed × single profile), lalu rerun top-1
        pemenang pada full ``seeds × profiles`` (cfg_idx=``SEARCH_BEST_CFG_IDX``)."""
        best = run_random_search(
            out_root=out_root,
            runs_root=runs_root,
            n_trials=int(args.random_search_n),
            sampling_seed=int(args.random_search_seed),
            search_train_seed=int(args.search_train_seed),
            search_profile=str(args.search_profile),
            train_eps=fold_entry["train_episodes"],
            val_eps=fold_entry["val_episodes"],
            zarr_rel=args.zarr_path,
            checkpoint_every=args.checkpoint_every,
            dataloader_num_workers=args.dataloader_num_workers,
            py=py,
            train_py=train_py,
            cwd_train=cwd_train,
            apply_vram_limits_fn=apply_vram_limits,
            max_batch_size=args.search_max_batch_size,
            center_hparams=baseline_cfg,
            sigma=float(args.random_search_sigma),
            p_exact_baseline=float(args.random_search_p_exact_baseline),
            enable_early_stop=enable_early_stop,
            early_stop_rollout_every=args.early_stop_rollout_every,
        )
        if best is None:
            print(
                "[random_search] WARNING: tidak ada pemenang; melewati fase rerun top-1."
            )
            return

        winner_cfg: Dict[str, Any] = dict(best["hparams"])
        winner_cfg["cfg_idx"] = SEARCH_BEST_CFG_IDX
        winner_cfg = apply_vram_limits(winner_cfg, args.max_batch_size)
        print(
            f"\n>>> Rerun pemenang random search (cfg_idx={SEARCH_BEST_CFG_IDX}) "
            f"pada {len(args.seeds)} seeds × {len(args.profiles)} profiles "
            f"@ training.num_epochs={int(winner_cfg['training.num_epochs'])}.\n"
        )
        run_grid_for_configs([winner_cfg])

    if args.baseline_only:
        run_grid_for_configs([baseline_cfg])
    elif search_only:
        run_random_search_phase()
    else:
        run_grid_for_configs([baseline_cfg])
        run_random_search_phase()

    summarize_script = SCRIPT_DIR / "summarize.py"
    plot_script = SCRIPT_DIR / "plot_results.py"
    _csv_args: List[str] = (
        ["--results-csv", str(results_csv)] if args.results_csv else []
    )
    subprocess.run(
        [py, str(summarize_script), "--output-dir", str(out_root)] + _csv_args,
        check=False,
    )
    subprocess.run(
        [py, str(plot_script), "--output-dir", str(out_root)] + _csv_args,
        check=False,
    )


if __name__ == "__main__":
    main()
