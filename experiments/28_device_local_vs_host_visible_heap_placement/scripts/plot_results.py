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
    summary = _load("device_local_vs_host_visible_heap_placement_summary.csv")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["variant"], summary["gpu_ms_median"], color=["tab:blue", "tab:orange"])
    ax.set_title("Experiment 28: Dispatch GPU Time by Placement")
    ax.set_ylabel("GPU ms (median)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "heap_placement_dispatch_gpu_ms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["variant"], summary["end_to_end_ms_median"], color=["tab:blue", "tab:orange"])
    ax.set_title("Experiment 28: End-to-End Time by Placement")
    ax.set_ylabel("End-to-end ms (median)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "heap_placement_end_to_end_ms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["variant"], summary["gbps_median"], color=["tab:green", "tab:red"])
    ax.set_title("Experiment 28: Effective Dispatch GB/s by Placement")
    ax.set_ylabel("GB/s (median)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "heap_placement_dispatch_gbps.png", dpi=150)
    plt.close(fig)

    print(f"[ok] Wrote Experiment 28 charts to {CHARTS_DIR}.")


if __name__ == "__main__":
    main()
