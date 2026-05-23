#!/usr/bin/env python3
"""Hyperband (Li et al., 2018; https://arxiv.org/pdf/1603.06560).

Selaras parameter API **KerasTuner** ``Hyperband``
(https://keras.io/keras_tuner/api/tuners/hyperband/):

- ``max_epochs`` → ``R`` (``--hyperband-max-epochs``)
- ``factor`` → ``eta`` (``--hyperband-eta``, default 3)
- ``hyperband_iterations`` → berapa kali mengulang seluruh algoritme
  (bracket ``s_max … s_min`` dengan sampel konfigurasi baru)

Implementasi mengikuti **Algoritma 1** pada paper:

::

    input: R, eta (default eta = 3)
    initialization: s_max = floor(log_eta(R)), B = (s_max + 1) * R

    for s in {s_max, s_max-1, ..., 0}:
        n = ceil((B / R) * eta^s / (s + 1))
        r = R * eta^(-s)
        # begin SuccessiveHalving with (n, r) inner loop
        T = get_hyperparameter_configuration(n)
        for i in {0, ..., s}:
            n_i = floor(n * eta^(-i))
            r_i = r * eta^i
            L = {run_then_return_val_loss(t, r_i) : t in T}
            T = top_k(T, L, floor(n_i / eta))
        end
    end
    return configuration with the smallest intermediate loss seen so far

Modul ini bersifat berkas-tunggal (single-file) — dipakai langsung oleh
``scripts/run_experiment.py``. Resource = jumlah epoch training FlowPolicy.
Sinyal "val_loss" antar-rung diambil dari ``training_final.json.val_loss_final``
yang ditulis oleh ``FlowPolicy/train.py`` di akhir setiap pelatihan.

Catatan deviasi praktis (didokumentasikan oleh paper, Section 5):
**reuse of trained weights between rungs**. Setelah rung ke-``i`` selesai,
folder run config yang lolos top-k dipertahankan; rung ke-``i+1`` melanjutkan
training dari ``latest.ckpt`` (``training.resume=true``) dengan
``training.num_epochs = r_{i+1} - r_i`` (delta) — bukan absolut, karena loop
training ``for local_epoch_idx in range(num_epochs)`` di ``train.py``
berjalan relatif terhadap kondisi saat resume. Ini menghemat banyak compute
dibanding melatih ulang dari awal pada setiap rung.

State persisten: ``<output-dir>/hyperband_state.json`` (lihat ``_load_state``
dan ``_save_state``). Eksekusi yang terputus dapat dilanjutkan: rung/config
yang sudah ada ``val_loss`` di state akan dilewati.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_constants import (  # noqa: E402
    CSV_HPARAM_KEYS,
    HYPERBAND_CFG_IDX_BASE,
    HYPERBAND_SAMPLING_BASELINE_ANCHORED,
    append_kitchen_policy_hparam_overrides,
    baseline_search_center,
    compute_horizon,
    sample_configs_hyperband,
)


# ---------------------------------------------------------------------------
# Penghitungan bracket sesuai Algoritma 1
# ---------------------------------------------------------------------------


@dataclass
class Rung:
    i: int
    n_i: int
    r_i: int  # dibulatkan ke integer epoch (ceil agar paling tidak melatih sebanyak r_i)


@dataclass
class Bracket:
    s: int
    n: int
    r: int  # dibulatkan integer
    rungs: List[Rung] = field(default_factory=list)


def compute_brackets(
    R: int,
    eta: int,
    *,
    s_min: int = 0,
    s_max: Optional[int] = None,
) -> List[Bracket]:
    """Bangun daftar bracket Hyperband untuk ``s in [s_max, ..., s_min]``.

    Mengikuti **kode referensi Jamieson** (coauthor paper,
    https://homes.cs.washington.edu/~jamieson/hyperband.html) yang
    mereproduksi Tabel 1 paper persis:

    - ``s_max_native = floor(log_eta(R))``
    - ``B = (s_max_native + 1) * R``
    - ``n = ceil( int(B / R / (s + 1)) * eta^s )`` — perhatikan ``int(...)``
      (integer division / floor) dilakukan SEBELUM perkalian ``eta^s``.
      Pseudo-code Algorithm 1 di paper menulis
      ``n = ceil((B/R) * eta^s / (s+1))`` tetapi Tabel 1 dan implementasi
      referensi memakai bentuk integer-divide-dulu ini.
    - ``r = R * eta^(-s)``
    - Rung ``i in {0, ..., s}``: ``n_i = floor(n * eta^(-i))``,
      ``r_i = r * eta^i``.

    ``s_max`` dapat di-cap di bawah ``s_max_native`` untuk membatasi waktu.
    ``B`` tetap memakai ``s_max_native`` (anggaran teoritis penuh) walaupun
    ``s_max`` dipotong — ini menjaga proporsi ``n_s`` per bracket konsisten
    dengan Hyperband asli yang dijalankan penuh.
    """
    if R <= 0:
        raise ValueError(f"R harus > 0, dapat {R}")
    if eta < 2:
        raise ValueError(f"eta harus >= 2, dapat {eta}")
    s_max_native = int(math.floor(math.log(R) / math.log(eta)))
    if s_max is None:
        s_max = s_max_native
    s_max = int(min(max(0, s_max), s_max_native))
    s_min = int(min(max(0, s_min), s_max))
    B = (s_max_native + 1) * R

    brackets: List[Bracket] = []
    for s in range(s_max, s_min - 1, -1):
        # Reference Jamieson:
        #   n = int(ceil(int(B / max_iter / (s+1)) * eta**s))
        # ``int(...)`` di Python untuk float positif sama dengan floor.
        n = int(math.ceil(int(B / R / (s + 1)) * (eta ** s)))
        r = R * (eta ** (-s))
        rungs: List[Rung] = []
        for i in range(s + 1):
            n_i = int(math.floor(n * (eta ** (-i))))
            r_i = r * (eta ** i)
            # Pembulatan epoch: rung terakhir = R persis agar config penyelamat
            # benar-benar dilatih sampai resource maksimum.
            if i == s:
                r_i_int = int(R)
            else:
                r_i_int = max(1, int(round(r_i)))
            n_i = max(1, n_i)
            rungs.append(Rung(i=i, n_i=n_i, r_i=r_i_int))
        brackets.append(Bracket(s=s, n=n, r=max(1, int(round(r))), rungs=rungs))
    return brackets


# ---------------------------------------------------------------------------
# Helper override Hydra (mirip ``run_experiment.build_train_overrides`` tapi
# disesuaikan untuk Hyperband: tanpa CSV row, fixed search seed+profile,
# resume training inkremental antar-rung).
# ---------------------------------------------------------------------------


def _build_train_overrides_hb(
    cfg: Dict[str, Any],
    *,
    seed: int,
    profile: str,
    train_eps: List[int],
    val_eps: List[int],
    run_dir: pathlib.Path,
    zarr_rel: str,
    resume_training: bool,
    delta_num_epochs: int,
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
        f"training.num_epochs={int(delta_num_epochs)}",
    ]
    append_kitchen_policy_hparam_overrides(odl, cfg)
    return odl


def _read_val_loss_final(run_dir: pathlib.Path) -> Optional[float]:
    p = run_dir / "training_final.json"
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return None
    v = d.get("val_loss_final")
    if v is None:
        return None
    try:
        vf = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(vf) or math.isinf(vf):
        return None
    return vf


def _run_dir_for_cfg(
    runs_root: pathlib.Path, cfg_idx: int, seed: int, profile: str
) -> pathlib.Path:
    return runs_root / f"hb_cfg{int(cfg_idx)}_seed{int(seed)}_{profile}"


def _split_key(seed: int, profile: str) -> str:
    return f"seed{int(seed)}_{profile}"


def _get_split_epoch_trained(cstate: Dict[str, Any], seed: int, profile: str) -> int:
    splits = cstate.get("split_epoch_trained") or {}
    return int(splits.get(_split_key(seed, profile), 0))


def _set_split_epoch_trained(
    cstate: Dict[str, Any], seed: int, profile: str, epoch_trained: int
) -> None:
    if "split_epoch_trained" not in cstate:
        cstate["split_epoch_trained"] = {}
    cstate["split_epoch_trained"][_split_key(seed, profile)] = int(epoch_trained)


def _aggregate_val_loss(values: List[Optional[float]]) -> Optional[float]:
    """Rata-rata val_loss lintas seed × profile (gagal/NaN diabaikan)."""
    good: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(vf) or math.isinf(vf):
            continue
        good.append(vf)
    if not good:
        return None
    return float(np.mean(good))


def _rung_eval_complete(
    ev: Optional[Dict[str, Any]],
    search_seeds: List[int],
    search_profiles: List[str],
) -> bool:
    if ev is None or ev.get("val_loss") is None:
        return False
    by_split = {
        (_split_key(int(s["seed"]), str(s["profile"]))): s
        for s in ev.get("by_split", [])
    }
    for seed in search_seeds:
        for profile in search_profiles:
            if _split_key(seed, profile) not in by_split:
                return False
            if by_split[_split_key(seed, profile)].get("val_loss") is None:
                return False
    return True


def _remove_cfg_run_dirs(runs_root: pathlib.Path, cfg_idx: int) -> None:
    """Hapus semua folder run untuk cfg_idx (semua seed × profile + legacy path)."""
    for pattern in (
        f"hb_cfg{int(cfg_idx)}_seed*",
        f"hb_cfg{int(cfg_idx)}",
    ):
        for p in runs_root.glob(pattern):
            if p.is_dir():
                try:
                    shutil.rmtree(p)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# State persisten (hyperband_state.json)
# ---------------------------------------------------------------------------


def _state_path(out_root: pathlib.Path) -> pathlib.Path:
    return out_root / "hyperband_state.json"


def _load_state(out_root: pathlib.Path) -> Optional[Dict[str, Any]]:
    p = _state_path(out_root)
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _save_state(out_root: pathlib.Path, state: Dict[str, Any]) -> None:
    p = _state_path(out_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, p)


def _bracket_state_from_sample(br: Bracket, cfgs: List[Dict[str, Any]]) -> Dict[str, Any]:
    configs_state: List[Dict[str, Any]] = []
    for cfg in cfgs:
        configs_state.append(
            {
                "cfg_idx": int(cfg["cfg_idx"]),
                "hparams": {k: cfg[k] for k in CSV_HPARAM_KEYS if k in cfg},
                    "epoch_trained": 0,
                    "split_epoch_trained": {},
                    "evaluations": [],
                    "active": True,
                }
            )
    return {
        "s": br.s,
        "n": br.n,
        "r": br.r,
        "rungs": [{"i": rg.i, "n_i": rg.n_i, "r_i": rg.r_i} for rg in br.rungs],
        "configs": configs_state,
        "completed_rungs": [],
        "survivors_final": [],
    }


def _iterations_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalisasi state v5 (``brackets`` top-level) → daftar iterasi."""
    if "iterations" in state and state["iterations"]:
        return list(state["iterations"])
    return [{"iteration": 0, "brackets": list(state.get("brackets", []))}]


def _build_initial_state(
    *,
    R: int,
    eta: int,
    s_min: int,
    s_max: int,
    hyperband_iterations: int,
    sampling: str,
    sampling_seed: int,
    search_seeds: List[int],
    search_profiles: List[str],
    brackets: List[Bracket],
    iterations_sampled: List[List[List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    """``iterations_sampled[it]`` = daftar config per bracket untuk iterasi ``it``."""
    iterations: List[Dict[str, Any]] = []
    for it, sampled_per_bracket in enumerate(iterations_sampled):
        bracket_states = [
            _bracket_state_from_sample(br, cfgs)
            for br, cfgs in zip(brackets, sampled_per_bracket)
        ]
        iterations.append({"iteration": int(it), "brackets": bracket_states})
    return {
        "version": 6,
        "algorithm": "hyperband",
        "R": int(R),
        "eta": int(eta),
        "s_min": int(s_min),
        "s_max": int(s_max),
        "hyperband_iterations": int(hyperband_iterations),
        "sampling": str(sampling),
        "baseline_center": baseline_search_center(),
        "sampling_seed": int(sampling_seed),
        "search_seeds": [int(s) for s in search_seeds],
        "search_profiles": [str(p) for p in search_profiles],
        "iterations": iterations,
        "brackets": iterations[0]["brackets"],
        "best": None,
        "all_evaluations": [],
    }


# ---------------------------------------------------------------------------
# Core: jalankan satu evaluasi (train ke target epoch via subprocess train.py)
# ---------------------------------------------------------------------------


def _evaluate_config_at_rung(
    *,
    cfg: Dict[str, Any],
    target_epoch: int,
    already_trained: int,
    run_dir: pathlib.Path,
    py: str,
    train_py: pathlib.Path,
    cwd_train: str,
    seed: int,
    profile: str,
    train_eps: List[int],
    val_eps: List[int],
    zarr_rel: str,
    checkpoint_every: int,
    dataloader_num_workers: int,
) -> Tuple[Optional[float], int, int]:
    """Latih ``cfg`` hingga ``target_epoch`` total (inkremental dari
    ``already_trained``). Kembalikan ``(val_loss, returncode, epoch_trained)``.

    Implementasi: hapus ``training_final.json`` lama agar train.py tidak
    mengira sudah selesai; set ``training.resume=true`` jika sudah ada ckpt;
    ``training.num_epochs = target_epoch - already_trained``.
    """
    delta = int(target_epoch) - int(already_trained)
    if delta <= 0:
        # Sudah lebih dari target — pakai val_loss yang ada.
        v = _read_val_loss_final(run_dir)
        return v, 0, int(already_trained)

    run_dir.mkdir(parents=True, exist_ok=True)

    # Hapus training_final.json lama agar train.py melanjutkan training.
    tf = run_dir / "training_final.json"
    if tf.is_file():
        try:
            tf.unlink()
        except Exception:
            pass

    ckpt = run_dir / "checkpoints" / "latest.ckpt"
    resume = bool(ckpt.is_file())

    overrides = _build_train_overrides_hb(
        cfg,
        seed=seed,
        profile=profile,
        train_eps=train_eps,
        val_eps=val_eps,
        run_dir=run_dir,
        zarr_rel=zarr_rel,
        resume_training=resume,
        delta_num_epochs=delta,
        checkpoint_every=checkpoint_every,
        dataloader_num_workers=dataloader_num_workers,
    )
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")

    cmd = [py, str(train_py)] + overrides
    print("\n" + "-" * 72)
    print(
        f"[hyperband] train cfg_idx={cfg['cfg_idx']} seed={seed} profile={profile} "
        f"→ target r_i={target_epoch} (delta={delta} epoch, resume={resume})"
    )
    print("-" * 72)
    rc = subprocess.run(cmd, cwd=cwd_train, env=env).returncode
    v = _read_val_loss_final(run_dir)
    return v, int(rc), int(target_epoch) if rc == 0 else int(already_trained)


def _evaluate_config_all_splits(
    *,
    cfg: Dict[str, Any],
    cstate: Dict[str, Any],
    target_epoch: int,
    runs_root: pathlib.Path,
    py: str,
    train_py: pathlib.Path,
    cwd_train: str,
    search_seeds: List[int],
    search_profiles: List[str],
    train_eps: List[int],
    val_eps: List[int],
    zarr_rel: str,
    checkpoint_every: int,
    dataloader_num_workers: int,
) -> Tuple[Optional[float], int, List[Dict[str, Any]]]:
    """Latih satu konfigurasi di semua ``search_seeds × search_profiles``.

    Mengembalikan ``(mean_val_loss, max_returncode, by_split)``.
    """
    by_split: List[Dict[str, Any]] = []
    max_rc = 0
    for seed in search_seeds:
        for profile in search_profiles:
            run_dir = _run_dir_for_cfg(runs_root, int(cfg["cfg_idx"]), seed, profile)
            already = _get_split_epoch_trained(cstate, seed, profile)
            val_loss, rc, ep_trained = _evaluate_config_at_rung(
                cfg=cfg,
                target_epoch=target_epoch,
                already_trained=already,
                run_dir=run_dir,
                py=py,
                train_py=train_py,
                cwd_train=cwd_train,
                seed=int(seed),
                profile=str(profile),
                train_eps=train_eps,
                val_eps=val_eps,
                zarr_rel=zarr_rel,
                checkpoint_every=checkpoint_every,
                dataloader_num_workers=dataloader_num_workers,
            )
            _set_split_epoch_trained(cstate, seed, profile, ep_trained)
            by_split.append(
                {
                    "seed": int(seed),
                    "profile": str(profile),
                    "val_loss": val_loss,
                    "rc": int(rc),
                }
            )
            max_rc = max(max_rc, int(rc))
    agg = _aggregate_val_loss([s["val_loss"] for s in by_split])
    # epoch_trained global = min epoch di semua split (kompatibilitas state lama).
    if by_split:
        cstate["epoch_trained"] = min(
            _get_split_epoch_trained(cstate, s["seed"], s["profile"]) for s in by_split
        )
    return agg, max_rc, by_split


# ---------------------------------------------------------------------------
# Successive Halving (inner loop) + Hyperband (outer loop)
# ---------------------------------------------------------------------------


def _top_k_by_val_loss(
    cfg_idxs: List[int],
    losses: Dict[int, Optional[float]],
    k: int,
) -> List[int]:
    """Sort cfg_idxs ascending by val_loss (None / NaN -> +inf), return first k."""
    def keyfn(c: int) -> float:
        v = losses.get(c)
        if v is None:
            return float("inf")
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return float("inf")
        if math.isnan(vf) or math.isinf(vf):
            return float("inf")
        return vf

    ordered = sorted(cfg_idxs, key=keyfn)
    return ordered[: max(0, int(k))]


def _successive_halving_for_bracket(
    *,
    bracket_state: Dict[str, Any],
    out_root: pathlib.Path,
    state: Dict[str, Any],
    runs_root: pathlib.Path,
    py: str,
    train_py: pathlib.Path,
    cwd_train: str,
    search_seeds: List[int],
    search_profiles: List[str],
    train_eps: List[int],
    val_eps: List[int],
    zarr_rel: str,
    checkpoint_every: int,
    dataloader_num_workers: int,
    eta: int,
) -> None:
    """Eksekusi SH untuk satu bracket (mengubah ``bracket_state`` in-place)."""
    n_splits = len(search_seeds) * len(search_profiles)
    s = int(bracket_state["s"])
    rungs = bracket_state["rungs"]

    # Daftar cfg_idx aktif saat ini (yang lolos sampai rung saat ini).
    active_cfg_idxs: List[int] = [
        c["cfg_idx"] for c in bracket_state["configs"] if c.get("active", True)
    ]

    cfg_by_idx: Dict[int, Dict[str, Any]] = {
        c["cfg_idx"]: c for c in bracket_state["configs"]
    }

    for rung in rungs:
        i = int(rung["i"])
        r_i = int(rung["r_i"])
        n_i_planned = int(rung["n_i"])

        # Lewati jika sudah pernah selesai sepenuhnya.
        completed = set(int(x) for x in bracket_state.get("completed_rungs", []))
        if i in completed:
            # Tetap update active_cfg_idxs dari evaluations.
            losses_i: Dict[int, Optional[float]] = {}
            for ci in active_cfg_idxs:
                cstate = cfg_by_idx[ci]
                ev = next(
                    (e for e in cstate["evaluations"] if int(e["i"]) == i), None
                )
                losses_i[ci] = ev.get("val_loss") if ev else None
            # Cull untuk lanjut ke rung berikut (jika ada).
            if i < s:
                k_keep = int(math.floor(n_i_planned / eta))
                survivors = _top_k_by_val_loss(active_cfg_idxs, losses_i, k_keep)
                for ci in active_cfg_idxs:
                    if ci not in survivors:
                        cfg_by_idx[ci]["active"] = False
                active_cfg_idxs = survivors
            continue

        print("\n" + "=" * 72)
        print(
            f"[hyperband] Bracket s={s}, rung i={i}/{s} | "
            f"n_i={len(active_cfg_idxs)} configs → r_i={r_i} epoch | "
            f"eval={n_splits} run/config (seeds×profiles, mean val_loss)"
        )
        print("=" * 72)

        losses_i = {}
        for ci in active_cfg_idxs:
            cstate = cfg_by_idx[ci]
            # Apakah evaluasi rung-i sudah ada (resume mesin mati)?
            ev_existing = next(
                (e for e in cstate["evaluations"] if int(e["i"]) == i), None
            )
            if _rung_eval_complete(ev_existing, search_seeds, search_profiles):
                print(
                    f"[skip] cfg_idx={ci}: rung i={i} sudah dievaluasi "
                    f"(mean val_loss={ev_existing['val_loss']:.6f}, "
                    f"{len(ev_existing.get('by_split', []))} split)"
                )
                losses_i[ci] = ev_existing["val_loss"]
                continue

            cfg = {k: cstate["hparams"][k] for k in cstate["hparams"]}
            cfg["cfg_idx"] = ci

            val_loss, rc, by_split = _evaluate_config_all_splits(
                cfg=cfg,
                cstate=cstate,
                target_epoch=r_i,
                runs_root=runs_root,
                py=py,
                train_py=train_py,
                cwd_train=cwd_train,
                search_seeds=search_seeds,
                search_profiles=search_profiles,
                train_eps=train_eps,
                val_eps=val_eps,
                zarr_rel=zarr_rel,
                checkpoint_every=checkpoint_every,
                dataloader_num_workers=dataloader_num_workers,
            )

            ev_new = {
                "i": i,
                "r_i": r_i,
                "val_loss": val_loss,
                "by_split": by_split,
                "rc": int(rc),
            }
            # Hapus eval lama untuk rung-i yang masih NULL (gagal), lalu append.
            cstate["evaluations"] = [
                e for e in cstate["evaluations"] if int(e["i"]) != i
            ] + [ev_new]
            losses_i[ci] = val_loss

            # Catat ke all_evaluations (untuk pemilihan "smallest intermediate loss").
            state["all_evaluations"].append(
                {
                    "bracket_s": s,
                    "rung_i": i,
                    "r_i": r_i,
                    "cfg_idx": ci,
                    "val_loss": val_loss,
                }
            )
            _save_state(out_root, state)

        # Tandai rung selesai.
        bracket_state["completed_rungs"] = sorted(
            set(int(x) for x in bracket_state.get("completed_rungs", [])) | {i}
        )

        # Cull: top floor(n_i / eta).
        if i < s:
            k_keep = int(math.floor(n_i_planned / eta))
            survivors = _top_k_by_val_loss(active_cfg_idxs, losses_i, k_keep)
            for ci in active_cfg_idxs:
                if ci not in survivors:
                    cfg_by_idx[ci]["active"] = False
                    _remove_cfg_run_dirs(runs_root, ci)
            active_cfg_idxs = survivors
            print(
                f"[hyperband] cull: top-{k_keep} survivor = {survivors}"
            )
        else:
            # Rung terakhir (i == s): tidak ada cull lagi.
            bracket_state["survivors_final"] = list(active_cfg_idxs)
            print(
                f"[hyperband] bracket s={s} selesai. survivors_final={active_cfg_idxs}"
            )

        _save_state(out_root, state)


def _pick_best_from_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pilih konfigurasi dengan ``val_loss`` TERKECIL dari seluruh
    ``all_evaluations`` lintas semua bracket dan rung — persis seperti
    paper Algorithm 1: "Configuration with the smallest intermediate loss
    seen so far".

    Catatan: pemenang BISA berasal dari rung dengan ``r_i < R``. Itu sesuai
    paper. Untuk evaluasi akhir yang adil, ``run_experiment.py`` melatih ulang
    pemenang dari scratch (atau lanjut dari ckpt-nya hingga R epoch penuh)
    pada 3 seeds × 2 profiles.
    """
    if not state["all_evaluations"]:
        return None

    def keyfn(e: Dict[str, Any]) -> float:
        v = e.get("val_loss")
        if v is None:
            return float("inf")
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return float("inf")
        if math.isnan(vf) or math.isinf(vf):
            return float("inf")
        return vf

    ordered = sorted(state["all_evaluations"], key=keyfn)
    best_ev = ordered[0]
    if keyfn(best_ev) == float("inf"):
        return None

    cfg_idx = int(best_ev["cfg_idx"])
    for iter_state in _iterations_from_state(state):
        for br in iter_state["brackets"]:
            for cstate in br["configs"]:
                if int(cstate["cfg_idx"]) == cfg_idx:
                    return {
                        "cfg_idx": cfg_idx,
                        "hparams": dict(cstate["hparams"]),
                        "val_loss": float(best_ev["val_loss"]),
                        "r_i": int(best_ev["r_i"]),
                        "bracket_s": int(best_ev["bracket_s"]),
                    }
    return None


# ---------------------------------------------------------------------------
# Entry: run_hyperband
# ---------------------------------------------------------------------------


def run_hyperband(
    *,
    out_root: pathlib.Path,
    runs_root: pathlib.Path,
    R: int,
    eta: int,
    s_min: int,
    s_max: Optional[int],
    hyperband_iterations: int = 1,
    sampling: str = HYPERBAND_SAMPLING_BASELINE_ANCHORED,
    sampling_seed: int,
    search_seeds: List[int],
    search_profiles: List[str],
    train_eps: List[int],
    val_eps: List[int],
    zarr_rel: str,
    checkpoint_every: int,
    dataloader_num_workers: int,
    py: str,
    train_py: pathlib.Path,
    cwd_train: str,
    apply_vram_limits_fn,
    max_batch_size: int,
) -> Optional[Dict[str, Any]]:
    """Jalankan Hyperband (Algorithm 1 paper) dan kembalikan pemenang.

    Setiap trial dievaluasi pada **semua** ``search_seeds × search_profiles``
    (default 3 seed × 2 profil: ``standard`` = data ter-augmentasi,
    ``minimal`` = tanpa augmentasi). Sinyal antar-rung = **rata-rata**
    ``val_loss`` lintas split tersebut.

    Return: dict ``{cfg_idx, hparams, val_loss, r_i, bracket_s}`` atau ``None``
    jika tidak ada evaluasi sukses.
    """
    search_seeds = [int(s) for s in search_seeds]
    search_profiles = [str(p) for p in search_profiles]
    if not search_seeds or not search_profiles:
        raise ValueError("search_seeds dan search_profiles tidak boleh kosong.")
    out_root = pathlib.Path(out_root).resolve()
    runs_root = pathlib.Path(runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    hyperband_iterations = max(1, int(hyperband_iterations))
    brackets = compute_brackets(R, eta, s_min=s_min, s_max=s_max)
    if not brackets:
        raise RuntimeError("compute_brackets mengembalikan daftar kosong.")

    # Resume jika state ada dan parameter cocok; jika tidak, buat ulang.
    state = _load_state(out_root)
    reuse_state = False
    if state is not None:
        same = (
            int(state.get("R", -1)) == int(R)
            and int(state.get("eta", -1)) == int(eta)
            and int(state.get("s_min", -1)) == int(s_min)
            and int(state.get("s_max", -1)) == int(brackets[0].s)
            and int(state.get("hyperband_iterations", 1)) == int(hyperband_iterations)
            and str(state.get("sampling", HYPERBAND_SAMPLING_BASELINE_ANCHORED))
            == str(sampling)
            and int(state.get("sampling_seed", -1)) == int(sampling_seed)
            and list(state.get("search_seeds", [])) == search_seeds
            and list(state.get("search_profiles", [])) == search_profiles
        )
        if same:
            reuse_state = True
        else:
            print(
                "[hyperband] parameter berubah vs hyperband_state.json — "
                "membuat state baru (file lama tetap di disk)."
            )

    if not reuse_state:
        print(
            f"[hyperband] Sampling: {sampling} "
            f"(pusat=baseline Franka Kitchen, bukan cold start)"
        )
        if str(sampling) == HYPERBAND_SAMPLING_BASELINE_ANCHORED:
            print(
                "[hyperband] Trial #0 tiap bracket = baseline persis; "
                "sisanya = tweak lokal ±1 di SEARCH_SPACE."
            )
        print(
            f"[hyperband] Evaluasi per trial: {len(search_seeds)} seeds × "
            f"{len(search_profiles)} profiles = "
            f"{len(search_seeds) * len(search_profiles)} training run "
            f"(seeds={search_seeds}, profiles={search_profiles})"
        )
        iterations_sampled: List[List[List[Dict[str, Any]]]] = []
        next_cfg_idx = int(HYPERBAND_CFG_IDX_BASE)
        for it in range(hyperband_iterations):
            rng_it = np.random.RandomState(int(sampling_seed) + it * 10007)
            sampled_per_bracket: List[List[Dict[str, Any]]] = []
            for br in brackets:
                cfgs = sample_configs_hyperband(
                    rng_it,
                    br.n,
                    base_cfg_idx=next_cfg_idx,
                    sampling=sampling,
                )
                cfgs = [apply_vram_limits_fn(c, max_batch_size) for c in cfgs]
                sampled_per_bracket.append(cfgs)
                next_cfg_idx += br.n
            iterations_sampled.append(sampled_per_bracket)
        state = _build_initial_state(
            R=R,
            eta=eta,
            s_min=s_min,
            s_max=int(brackets[0].s),
            hyperband_iterations=hyperband_iterations,
            sampling=sampling,
            sampling_seed=sampling_seed,
            search_seeds=search_seeds,
            search_profiles=search_profiles,
            brackets=brackets,
            iterations_sampled=iterations_sampled,
        )
        _save_state(out_root, state)

    # Eksekusi: tiap iterasi (Keras ``hyperband_iterations``), bracket s_max → s_min.
    for iter_state in _iterations_from_state(state):
        it = int(iter_state.get("iteration", 0))
        print(
            f"\n[hyperband] Iterasi {it + 1}/{hyperband_iterations} "
            f"(hyperband_iterations, KerasTuner)"
        )
        for bracket_state in iter_state["brackets"]:
            if bracket_state.get("survivors_final"):
                print(
                    f"[skip] iter={it} bracket s={bracket_state['s']} sudah selesai "
                    f"(survivors_final={bracket_state['survivors_final']})"
                )
                continue
            _successive_halving_for_bracket(
                bracket_state=bracket_state,
                out_root=out_root,
                state=state,
                runs_root=runs_root,
                py=py,
                train_py=train_py,
                cwd_train=cwd_train,
                search_seeds=search_seeds,
                search_profiles=search_profiles,
                train_eps=train_eps,
                val_eps=val_eps,
                zarr_rel=zarr_rel,
                checkpoint_every=checkpoint_every,
                dataloader_num_workers=dataloader_num_workers,
                eta=eta,
            )
        state["brackets"] = iter_state["brackets"]
        _save_state(out_root, state)

    best = _pick_best_from_state(state)
    state["best"] = best
    _save_state(out_root, state)

    if best is None:
        print(
            "[hyperband] WARNING: tidak ada evaluasi sukses; tidak ada pemenang."
        )
    else:
        print("\n" + "#" * 72)
        print(
            f"[hyperband] PEMENANG: cfg_idx={best['cfg_idx']} "
            f"(bracket s={best['bracket_s']}, r_i={best['r_i']}, "
            f"val_loss={best['val_loss']:.6f})"
        )
        print(json.dumps(best["hparams"], indent=2, default=str))
        print("#" * 72 + "\n")
    return best
