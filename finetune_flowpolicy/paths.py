"""Path helpers so kode di paket ini bisa import dari `FlowPolicy/` & `ReinFlow/`
tanpa harus install editable. Import file ini SEBELUM import dari kedua paket itu.
"""
from __future__ import annotations

import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_FLOWPOLICY_ROOT = _REPO_ROOT / "FlowPolicy"
_REINFLOW_ROOT = _REPO_ROOT / "ReinFlow"


def setup_sys_path() -> None:
    """Tambahkan FlowPolicy/ dan ReinFlow/ ke sys.path (idempoten)."""
    for p in (str(_FLOWPOLICY_ROOT), str(_REINFLOW_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # ReinFlow logging dir, dipakai default oleh cfg ReinFlow lain
    os.environ.setdefault("REINFLOW_LOG_DIR", str(_REPO_ROOT / "outputs"))
    os.environ.setdefault("REINFLOW_DATA_DIR", str(_REPO_ROOT / "data"))


def repo_root() -> pathlib.Path:
    return _REPO_ROOT


def flowpolicy_root() -> pathlib.Path:
    return _FLOWPOLICY_ROOT


def reinflow_root() -> pathlib.Path:
    return _REINFLOW_ROOT


# auto-setup on import
setup_sys_path()
