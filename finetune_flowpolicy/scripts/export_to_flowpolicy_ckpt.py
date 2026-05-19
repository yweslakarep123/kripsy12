"""Konversi checkpoint PPO ReinFlow ('*.pt') -> format FlowPolicy ('*.ckpt').

`TrainPPOFlowPolicyAgent` sudah secara otomatis menulis ``last_flowpolicy.ckpt`` &
``best_flowpolicy.ckpt`` di samping ``.pt`` ReinFlow setiap kali ``save_model`` dipanggil.
Script ini berguna kalau Anda punya checkpoint ReinFlow `.pt` yang belum di-convert
(mis. checkpoint lama tanpa hook ``_emit_flowpolicy_ckpt``), atau ingin override custom.

Contoh::

    python finetune_flowpolicy/scripts/export_to_flowpolicy_ckpt.py \
        --pt-ckpt outputs/.../checkpoint/best.pt \
        --src-ckpt outputs/baseline_seed101_standard/checkpoints/latest-001.ckpt \
        --out outputs/.../checkpoint/best_flowpolicy.ckpt

Setelah itu langsung evaluasi::

    python FlowPolicy/infer_kitchen.py \
        --checkpoint outputs/.../checkpoint/best_flowpolicy.ckpt \
        --metrics-json outputs/.../eval.json
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

_THIS = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from finetune_flowpolicy import paths as _paths  # noqa: F401  setup sys.path
from finetune_flowpolicy.utils.ckpt_io import export_to_flowpolicy_ckpt  # noqa: E402


def _extract_unet_state_dict_from_pt(pt_payload: dict) -> dict:
    """Cari state-dict ConditionalUnet1D di payload PPO ReinFlow.

    Format yang didukung:
    1) payload["model"] berisi ``actor_ft.policy.unet.*``
       (default TrainPPOFlowAgent.save_model).
    2) payload["policy"] berisi ``network.*`` -- bisa muncul kalau pakai
       ``only_save_policy_network=True``. Dalam kasus kita,
       ``actor_ft.policy.unet.*`` -> ``network.unet.*`` (prefix di-rename).
    """
    candidates: list[dict] = []
    if isinstance(pt_payload, dict):
        if "model" in pt_payload and isinstance(pt_payload["model"], dict):
            candidates.append(pt_payload["model"])
        if "policy" in pt_payload and isinstance(pt_payload["policy"], dict):
            candidates.append(pt_payload["policy"])

    for sd in candidates:
        # cari prefix 'actor_ft.policy.unet.'
        prefix_ft = "actor_ft.policy.unet."
        unet_sd = {k[len(prefix_ft):]: v for k, v in sd.items() if k.startswith(prefix_ft)}
        if unet_sd:
            return unet_sd
        # alternatif: 'network.unet.' setelah only_save_policy_network=True
        prefix_net = "network.unet."
        unet_sd = {k[len(prefix_net):]: v for k, v in sd.items() if k.startswith(prefix_net)}
        if unet_sd:
            return unet_sd
        # kalau seluruh sd memang sudah unet (langsung load_unet_state_dict-compatible)
        # heuristic: ada key 'diffusion_step_encoder.0.weight' atau 'final_conv.0.block.0.weight'
        unet_marker = "diffusion_step_encoder.0.weight"
        if unet_marker in sd:
            return dict(sd)

    raise RuntimeError(
        "Tidak menemukan state-dict ConditionalUnet1D di payload PPO. "
        "Pastikan checkpoint berisi key 'model' / 'policy' dengan prefix "
        "'actor_ft.policy.unet.' atau 'network.unet.'."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pt-ckpt",
        required=True,
        help="Path checkpoint PPO ReinFlow (.pt) yang akan diconvert.",
    )
    p.add_argument(
        "--src-ckpt",
        required=True,
        help="Path checkpoint FlowPolicy pretrained (.ckpt) sebagai sumber cfg + encoder + normalizer.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Path file .ckpt format FlowPolicy yang akan ditulis.",
    )
    args = p.parse_args()

    pt_path = pathlib.Path(args.pt_ckpt).resolve()
    src_path = pathlib.Path(args.src_ckpt).resolve()
    out_path = pathlib.Path(args.out).resolve()

    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)
    if not src_path.is_file():
        raise FileNotFoundError(src_path)

    payload = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    unet_sd = _extract_unet_state_dict_from_pt(payload)
    print(f"[export] extracted ConditionalUnet1D state_dict: {len(unet_sd)} keys")

    written = export_to_flowpolicy_ckpt(
        src_ckpt_path=str(src_path),
        new_unet_sd=unet_sd,
        out_path=str(out_path),
    )
    print(f"[export] wrote -> {written}")


if __name__ == "__main__":
    main()
