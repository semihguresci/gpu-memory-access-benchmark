#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"


def _load(path: str) -> pd.DataFrame:
    frame = pd.read_csv(TABLES_DIR / path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")
    return frame


def main() -> None:
    summary = _load("two_dimensional_locality_transpose_study_summary.csv")
    relative = _load("two_dimensional_locality_transpose_study_relative.csv")
    stability = _load("two_dimensional_locality_transpose_study_stability.csv")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["variant"], summary["gpu_ms_median"], color=["tab:blue", "tab:red", "tab:orange", "tab:green"])
    ax.set_title("Experiment 33: GPU Time by Variant")
    ax.set_ylabel("GPU ms (median)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "transpose_gpu_ms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(relative["variant"], relative["speedup_vs_naive_transpose"], color="tab:orange")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 33: Speedup vs naive_transpose")
    ax.set_ylabel("speedup factor")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "transpose_speedup_vs_naive.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(stability["variant"], stability["p95_to_median_gpu_ms"], color="tab:purple")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 33: Stability by Variant")
    ax.set_ylabel("p95 / median GPU ms")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "transpose_stability.png", dpi=150)
    plt.close(fig)

    print(f"[ok] Wrote Experiment 33 charts to {CHARTS_DIR}.")


if __name__ == "__main__":
    main()
