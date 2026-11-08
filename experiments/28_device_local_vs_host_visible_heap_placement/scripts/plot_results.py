#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Plot only tables bound to validated raw measurements.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analysis_integrity import (
    validate_analysis_provenance,
    validate_derived_table,
    validate_summary_table,
    write_plot_provenance,
)

EXPERIMENT_ID = "28_device_local_vs_host_visible_heap_placement"

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
    summary = _load("device_local_vs_host_visible_heap_placement_summary.csv")
    relative = _load("device_local_vs_host_visible_heap_placement_relative.csv")
    try:
        amortization = _load("device_local_vs_host_visible_reuse_amortization.csv")
    except (FileNotFoundError, ValueError):
        amortization = None
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    colors = [
        "tab:blue" if str(variant).startswith("device_local") else "tab:orange"
        for variant in summary["variant"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["variant"], summary["gpu_ms_median"], color=colors)
    ax.set_title("Experiment 28: Full GPU-Path Time by Placement and Reuse Group")
    ax.set_ylabel("GPU ms (median)")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "heap_placement_gpu_path_ms.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "heap_placement_gpu_path_ms.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["variant"], summary["end_to_end_ms_median"], color=colors)
    ax.set_title("Experiment 28: Harness Wall-Clock Time by Placement and Reuse Group")
    ax.set_ylabel("Harness wall-clock ms (median)")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "heap_placement_harness_wall_clock_ms.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "heap_placement_harness_wall_clock_ms.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["variant"], summary["gbps_median"], color=colors)
    ax.set_title("Experiment 28: Effective Full GPU-Path GB/s by Placement and Reuse Group")
    ax.set_ylabel("GB/s (median)")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "heap_placement_gpu_path_gbps.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "heap_placement_gpu_path_gbps.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    staged = relative[relative["variant"].str.startswith("device_local_staged")]
    for problem_size, group in staged.groupby("problem_size"):
        group = group.sort_values("chain_kernel_count")
        ax.plot(
            group["chain_kernel_count"],
            group["slowdown_vs_host_visible_gpu_path"],
            marker="o",
            linewidth=2.0,
            label=f"{int(problem_size):,} elements",
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log", base=2)
    ax.set_title("Experiment 28: Staged / Direct Time at Matched Kernel Counts")
    ax.set_xlabel("kernels between one upload and one readback (K, log2)")
    ax.set_ylabel("staged / direct median GPU path ms (same K)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "heap_placement_matched_kernel_count_ratio.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "heap_placement_matched_kernel_count_ratio.png")
    plt.close(fig)

    if amortization is not None and not amortization.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            amortization["chain_kernel_count"],
            amortization["direct_per_kernel_gpu_path_ms"],
            marker="o",
            linewidth=2.0,
            label="host_visible_direct per kernel",
        )
        ax.plot(
            amortization["chain_kernel_count"],
            amortization["staged_per_kernel_gpu_path_ms"],
            marker="s",
            linewidth=2.0,
            label="device_local_staged per kernel (upload+K kernels+readback)/K",
        )
        ax.set_xscale("log", base=2)
        ax.set_title("Experiment 28: Reuse Amortization Crossover")
        ax.set_xlabel("kernels between one upload and one readback (K, log2)")
        ax.set_ylabel("GPU path ms per kernel (median)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / "heap_placement_reuse_amortization.png", dpi=150)
        generated_charts.append(CHARTS_DIR / "heap_placement_reuse_amortization.png")
        plt.close(fig)

    print(f"[ok] Wrote Experiment 28 charts to {CHARTS_DIR}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
