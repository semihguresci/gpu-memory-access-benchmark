#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analysis_integrity import (
    validate_analysis_provenance,
    validate_derived_table,
    validate_summary_table,
    write_plot_provenance,
)

EXPERIMENT_ID = "46_dependent_load_latency"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"


def _load(path: str) -> pd.DataFrame:
    validate_derived_table(TABLES_DIR / path)
    frame = pd.read_csv(TABLES_DIR / path)
    validate_summary_table(frame, TABLES_DIR / path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")
    return frame


def main() -> None:
    generated_charts: list[Path | None] = []
    validate_analysis_provenance(TABLES_DIR, EXPERIMENT_ID)
    summary = _load("dependent_load_latency_summary.csv")
    latency = _load("dependent_load_latency_per_load_latency.csv")
    relative = _load("dependent_load_latency_relative.csv")
    stability = _load("dependent_load_latency_stability.csv")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in summary["variant"].unique():
        subset = summary[summary["variant"] == variant].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["gpu_ms_median"], marker="o", linewidth=2.0, label=variant)
    ax.set_title("Experiment 46: GPU Time by Element Count")
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("GPU ms (median)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "dependent_load_latency_ms_vs_size.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "dependent_load_latency_ms_vs_size.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in latency["variant"].unique():
        subset = latency[latency["variant"] == variant].sort_values("problem_size")
        ax.plot(
            subset["problem_size"],
            subset["ns_per_dependent_load_median"],
            marker="o",
            linewidth=2.0,
            label=variant,
        )
    ax.set_title("Experiment 46: Effective Dependent-Load Latency")
    ax.set_xlabel("pointer-table problem size (elements)")
    ax.set_ylabel("ns per dependent load (median)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "dependent_load_latency_ns_per_load_vs_size.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "dependent_load_latency_ns_per_load_vs_size.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in relative["variant"].unique():
        subset = relative[relative["variant"] == variant].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["slowdown_vs_single_chain"], marker="o", linewidth=2.0, label=variant)
    ax.set_title("Experiment 46: Slowdown vs chain_1 Baseline")
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("slowdown vs single chain (variant time / chain_1 time)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "dependent_load_latency_relative.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "dependent_load_latency_relative.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in stability["variant"].unique():
        subset = stability[stability["variant"] == variant].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["p95_to_median_gpu_ms"], marker="o", linewidth=2.0, label=variant)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 46: Stability by Element Count")
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("p95 / median GPU ms")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "dependent_load_latency_stability.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "dependent_load_latency_stability.png")
    plt.close(fig)

    print(f"[ok] Wrote Experiment 46 charts to {CHARTS_DIR}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
