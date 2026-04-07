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
    summary = _load("subgroup_scan_variants_summary.csv")
    relative = _load("subgroup_scan_variants_relative.csv")
    stability = _load("subgroup_scan_variants_stability.csv")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in summary["variant"].unique():
        subset = summary[summary["variant"] == variant].sort_values("items_per_thread")
        ax.plot(subset["items_per_thread"], subset["gpu_ms_median"], marker="o", linewidth=2.0, label=variant)
    ax.set_title("Experiment 31: GPU Time by Items per Thread")
    ax.set_xlabel("items per thread")
    ax.set_ylabel("GPU ms (median)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "subgroup_scan_gpu_ms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    subgroup = relative[relative["variant"].str.startswith("subgroup_block_scan")].sort_values("items_per_thread")
    ax.plot(subgroup["items_per_thread"], subgroup["speedup_vs_shared_block_scan"], marker="o", linewidth=2.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 31: subgroup_block_scan Speedup vs shared_block_scan")
    ax.set_xlabel("items per thread")
    ax.set_ylabel("speedup factor")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "subgroup_scan_speedup.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in stability["variant"].unique():
        subset = stability[stability["variant"] == variant].sort_values("items_per_thread")
        ax.plot(subset["items_per_thread"], subset["p95_to_median_gpu_ms"], marker="o", linewidth=2.0, label=variant)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 31: Stability by Items per Thread")
    ax.set_xlabel("items per thread")
    ax.set_ylabel("p95 / median GPU ms")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "subgroup_scan_stability.png", dpi=150)
    plt.close(fig)

    print(f"[ok] Wrote Experiment 31 charts to {CHARTS_DIR}.")


if __name__ == "__main__":
    main()
