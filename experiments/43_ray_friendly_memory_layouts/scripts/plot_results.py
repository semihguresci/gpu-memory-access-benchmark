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
    summary = _load("ray_friendly_memory_layouts_summary.csv")
    relative = _load("ray_friendly_memory_layouts_relative.csv")
    stability = _load("ray_friendly_memory_layouts_stability.csv")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in summary["variant"].unique():
        subset = summary[summary["variant"] == variant].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["gpu_ms_median"], marker="o", linewidth=2.0, label=variant)
    ax.set_title("Experiment 43: GPU Time by Primitive Count")
    ax.set_xlabel("problem size (primitives)")
    ax.set_ylabel("GPU ms (median)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ray_friendly_memory_layouts_ms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in summary["variant"].unique():
        subset = summary[summary["variant"] == variant].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["throughput_median"], marker="o", linewidth=2.0, label=variant)
    ax.set_title("Experiment 43: Throughput by Primitive Count")
    ax.set_xlabel("problem size (primitives)")
    ax.set_ylabel("throughput (elements/s, median)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ray_friendly_memory_layouts_throughput.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in relative["variant"].unique():
        subset = relative[relative["variant"] == variant].sort_values("problem_size")
        ax.plot(
            subset["problem_size"],
            subset["slowdown_vs_aos64_sequential"],
            marker="o",
            linewidth=2.0,
            label=variant,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 43: Slowdown vs aos64_sequential")
    ax.set_xlabel("problem size (primitives)")
    ax.set_ylabel("slowdown ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ray_friendly_memory_layouts_relative.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in stability["variant"].unique():
        subset = stability[stability["variant"] == variant].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["p95_to_median_gpu_ms"], marker="o", linewidth=2.0, label=variant)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 43: Stability by Primitive Count")
    ax.set_xlabel("problem size (primitives)")
    ax.set_ylabel("p95 / median GPU ms")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "ray_friendly_memory_layouts_stability.png", dpi=150)
    plt.close(fig)

    print(f"[ok] Wrote Experiment 43 charts to {CHARTS_DIR}.")


if __name__ == "__main__":
    main()
