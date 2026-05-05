#!/usr/bin/env python3
"""Agregasi results.csv → summary.csv (mean±std lintas seed)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(output_dir: Path) -> None:
    results_path = output_dir / "results.csv"
    if not results_path.is_file():
        print(f"Tidak ada {results_path}")
        return

    df = pd.read_csv(results_path)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("Tidak ada baris status=ok untuk diagregasi.")
        return

    metrics = [
        "success_rate_k1",
        "success_rate_k2",
        "success_rate_k3",
        "success_rate_k4",
        "mean_inference_latency_ms",
    ]
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    df["trade_off_computed"] = np.where(
        df["mean_inference_latency_ms"] > 1e-9,
        df["success_rate_k4"] / df["mean_inference_latency_ms"],
        np.nan,
    )

    gcols = ["cfg_idx", "profile", "fold"]
    agg_rows = []
    for keys, sub in df.groupby(gcols):
        if isinstance(keys, tuple):
            row = dict(zip(gcols, keys))
        else:
            row = {gcols[0]: keys}
        for m in metrics:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_std"] = float(sub[m].std(ddof=0))
        row["trade_off_mean"] = float(sub["trade_off_computed"].mean())
        row["trade_off_std"] = float(sub["trade_off_computed"].std(ddof=0))
        agg_rows.append(row)

    out = pd.DataFrame(agg_rows)
    out = out.sort_values("trade_off_mean", ascending=False)
    out_path = output_dir / "summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Ditulis {out_path} ({len(out)} baris)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default="outputs/experiment")
    repo_root = Path(__file__).resolve().parent.parent
    args = ap.parse_args()
    summarize((repo_root / args.output_dir).resolve())


if __name__ == "__main__":
    main()
