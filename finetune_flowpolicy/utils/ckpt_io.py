"""Load & convert checkpoint FlowPolicy <-> format intermediate untuk RL fine-tuning.

Format checkpoint FlowPolicy (`*.ckpt`):
    payload = {
        'cfg': OmegaConf,
        'state_dicts': {
            'model':       <FlowPolicy state_dict (full)>,
            'ema_model':   <FlowPolicy state_dict (full)>,
            'optimizer':   ...,
            'lr_scheduler':...,
        },
        'pickles': {'_output_dir':..., 'global_step':..., 'epoch':...},
    }

State-dict FlowPolicy berisi prefix-prefix:
    obs_encoder.*       -> FlowPolicyEncoder
    model.*             -> ConditionalUnet1D
    normalizer.*        -> LinearNormalizer
    mask_generator.*    -> LowdimMaskGenerator (tidak dipakai di RL)
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional, Tuple

import torch

import finetune_flowpolicy.paths  # noqa: F401  side-effect: setup sys.path


def _filter_prefix(state_dict: Dict[str, Any], prefix: str, strip: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    p = prefix if prefix.endswith(".") else prefix + "."
    for k, v in state_dict.items():
        if k.startswith(p):
            new_k = k[len(p):] if strip else k
            out[new_k] = v
    return out


def load_payload(ckpt_path: str | pathlib.Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Baca file .ckpt FlowPolicy (pickle_module=dill). Mengembalikan payload dict."""
    import dill

    p = pathlib.Path(ckpt_path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {p}")
    with open(p, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dicts" not in payload:
        raise ValueError(
            f"Format checkpoint tak dikenali (key={list(payload.keys()) if isinstance(payload, dict) else type(payload)})"
        )
    return payload


def select_full_state_dict(payload: Dict[str, Any], use_ema: bool = True) -> Dict[str, Any]:
    """Ambil FlowPolicy state_dict (EMA atau non-EMA)."""
    sds = payload["state_dicts"]
    key = "ema_model" if (use_ema and "ema_model" in sds) else "model"
    if key not in sds:
        raise KeyError(f"Tidak ada key {key} di state_dicts (tersedia: {list(sds.keys())})")
    return sds[key]


def split_flowpolicy_state_dict(full_sd: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Pisahkan state_dict FlowPolicy ke sub-modules.

    Returns:
        {
            'unet':      state_dict ConditionalUnet1D,
            'encoder':   state_dict FlowPolicyEncoder,
            'normalizer':state_dict LinearNormalizer,
        }
    """
    return {
        "unet": _filter_prefix(full_sd, "model"),
        "encoder": _filter_prefix(full_sd, "obs_encoder"),
        "normalizer": _filter_prefix(full_sd, "normalizer"),
    }


def load_flowpolicy_components(
    ckpt_path: str | pathlib.Path,
    *,
    use_ema: bool = True,
    map_location: str = "cpu",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Any]:
    """Convenience: load checkpoint dan kembalikan (unet_sd, encoder_sd, normalizer_sd, cfg).

    Cfg dikembalikan sebagai OmegaConf agar bisa diintrospeksi (shape_meta, dst).
    """
    payload = load_payload(ckpt_path, map_location=map_location)
    full = select_full_state_dict(payload, use_ema=use_ema)
    parts = split_flowpolicy_state_dict(full)
    return parts["unet"], parts["encoder"], parts["normalizer"], payload.get("cfg")


# ----------------------------------------------------------------------
# Exporter: tulis kembali ke format FlowPolicy *.ckpt
# ----------------------------------------------------------------------


def assemble_flowpolicy_state_dict(
    unet_sd: Dict[str, Any],
    encoder_sd: Dict[str, Any],
    normalizer_sd: Dict[str, Any],
    extra_sd: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Susun kembali state_dict FlowPolicy lengkap dengan prefix sesuai modul.

    extra_sd: dict {prefix: state_dict} untuk komponen lain (mis. mask_generator)
    yang ingin dipreservasi dari original checkpoint.
    """
    out: Dict[str, Any] = {}
    for k, v in unet_sd.items():
        out[f"model.{k}"] = v
    for k, v in encoder_sd.items():
        out[f"obs_encoder.{k}"] = v
    for k, v in normalizer_sd.items():
        out[f"normalizer.{k}"] = v
    if extra_sd:
        for prefix, sd in extra_sd.items():
            for k, v in sd.items():
                out[f"{prefix}.{k}"] = v
    return out


def export_to_flowpolicy_ckpt(
    src_ckpt_path: str | pathlib.Path,
    new_unet_sd: Dict[str, Any],
    out_path: str | pathlib.Path,
    *,
    new_encoder_sd: Optional[Dict[str, Any]] = None,
    new_normalizer_sd: Optional[Dict[str, Any]] = None,
    overwrite_ema_only: bool = True,
) -> pathlib.Path:
    """Tulis checkpoint baru dengan UNet hasil PPO ke format FlowPolicy.

    Encoder + normalizer + komponen lain dipertahankan dari ``src_ckpt_path``
    kecuali dikirim eksplisit lewat parameter.

    Args:
        src_ckpt_path: checkpoint pretrained sebagai sumber (untuk cfg + pickles + komponen non-trained).
        new_unet_sd: state_dict ConditionalUnet1D hasil fine-tuning (tanpa prefix `model.`).
        out_path: path file .ckpt tujuan.
        overwrite_ema_only: jika True, hanya `state_dicts.ema_model` & `state_dicts.model`
            yang ditimpa (lainnya dipertahankan); infer_kitchen.py default pakai ema_model.

    Returns:
        path file checkpoint yang ditulis.
    """
    import dill

    src = load_payload(src_ckpt_path, map_location="cpu")

    # state-dict template diambil dari ema_model (atau model) original
    template = select_full_state_dict(src, use_ema=True)
    # ekstrak key extra (selain unet/encoder/normalizer) untuk dipreservasi as-is.
    extras_flat: Dict[str, Any] = {}
    known_prefixes = ("model.", "obs_encoder.", "normalizer.")
    for k, v in template.items():
        if any(k.startswith(p) for p in known_prefixes):
            continue
        extras_flat[k] = v

    # encoder & normalizer: pakai param eksplisit kalau ada, kalau tidak ambil dari template
    if new_encoder_sd is None:
        new_encoder_sd = _filter_prefix(template, "obs_encoder")
    if new_normalizer_sd is None:
        new_normalizer_sd = _filter_prefix(template, "normalizer")

    rebuilt = assemble_flowpolicy_state_dict(
        unet_sd=new_unet_sd,
        encoder_sd=new_encoder_sd,
        normalizer_sd=new_normalizer_sd,
    )
    # tambahkan extras tanpa pengubahan prefix
    for k, v in extras_flat.items():
        rebuilt[k] = v

    payload = dict(src)
    payload["state_dicts"] = dict(src["state_dicts"])
    payload["state_dicts"]["ema_model"] = rebuilt
    payload["state_dicts"]["model"] = rebuilt
    # buang optimizer & scheduler agar checkpoint lebih ringkas (tidak relevan untuk inference)
    payload["state_dicts"].pop("optimizer", None)
    payload["state_dicts"].pop("lr_scheduler", None)

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        torch.save(payload, f, pickle_module=dill)
    return out_path
