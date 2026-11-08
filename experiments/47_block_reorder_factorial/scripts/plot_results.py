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

EXPERIMENT_ID = "47_block_reorder_factorial"

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
    summary = _load("block_reorder_factorial_summary.csv")
    effects = _load("block_reorder_factorial_effects.csv")
    residency = _load("block_reorder_factorial_residency.csv")
    stability = _load("block_reorder_factorial_stability.csv")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Factor effect ratios per block width at the largest and smallest sizes.
    fig, ax = plt.subplots(figsize=(10, 6))
    shown_sizes = sorted({float(size) for size in effects["problem_size"]})
    for problem_size in [shown_sizes[-1], shown_sizes[0]]:
        subset = effects[effects["problem_size"] == float(problem_size)].sort_values("block_width")
        ax.plot(subset["block_width"], subset["within_reversal_ratio"], marker="o", linewidth=2.0,
                label=f"within-reversal @ size {int(problem_size)}")
        ax.plot(subset["block_width"], subset["block_shuffle_ratio"], marker="s", linewidth=1.6, linestyle="--",
                label=f"block-shuffle @ size {int(problem_size)}")
        ax.plot(subset["block_width"], subset["combined_ratio"], marker="^", linewidth=1.6, linestyle=":",
                label=f"combined @ size {int(problem_size)}")
    ax.set_title("Experiment 47: Factor Effect Ratios by Block Width")
    ax.set_xlabel("block width (elements)")
    ax.set_ylabel("median GPU time ratio vs same-width sequential/ordered")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "block_reorder_factorial_effects_by_width.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "block_reorder_factorial_effects_by_width.png")
    plt.close(fig)

    # Use the smallest collected width, which depends on the device subgroup size.
    selected_width = int(summary["block_width"].min())
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in sorted(summary["variant"].unique()):
        if not variant.startswith(f"w{selected_width}_"):
            continue
        subset = summary[summary["variant"] == variant].sort_values("problem_size")
        style = {"color": None, "linestyle": "--"} if variant.endswith("_evicted") else {}
        ax.plot(subset["problem_size"], subset["gpu_ms_median"], marker="o", linewidth=1.8, label=variant, **style)
    ax.set_title(f"Experiment 47: GPU Time by Element Count (w{selected_width} variants)")
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("GPU ms (median)")
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "block_reorder_factorial_ms_vs_size_smallest_width.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "block_reorder_factorial_ms_vs_size_smallest_width.png")
    plt.close(fig)

    # Evicted-over-warm residency ratios by problem size per variant.
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in sorted(residency["evicted_variant"].unique()) if not residency.empty else []:
        subset = residency[residency["evicted_variant"] == variant].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["evicted_over_warm_ratio"], marker="o", linewidth=1.8, label=variant)
    if residency.empty:
        ax.text(0.5, 0.5, "No evicted residency rows collected", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        ax.set_title("Experiment 47: Evicted-over-Warm Residency Ratios")
        ax.set_xlabel("problem size (elements)")
        ax.set_ylabel("evicted / warm median GPU time")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "block_reorder_factorial_residency_ratio.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "block_reorder_factorial_residency_ratio.png")
    plt.close(fig)

    # Stability overview for all variants.
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in sorted(stability["variant"].unique()):
        subset = stability[stability["variant"] == variant].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["p95_to_median_gpu_ms"], marker="o", linewidth=1.4, label=variant)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 47: Stability by Element Count")
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("p95 / median GPU ms")
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "block_reorder_factorial_stability.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "block_reorder_factorial_stability.png")
    plt.close(fig)

    print(f"[ok] Wrote Experiment 47 charts to {CHARTS_DIR}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
