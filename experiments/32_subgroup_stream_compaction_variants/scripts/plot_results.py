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

EXPERIMENT_ID = "32_subgroup_stream_compaction_variants"

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
    summary = _load("subgroup_stream_compaction_variants_summary.csv")
    relative = _load("subgroup_stream_compaction_variants_relative.csv")
    stability = _load("subgroup_stream_compaction_variants_stability.csv")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in summary["variant"].unique():
        subset = summary[summary["variant"] == variant].sort_values("valid_ratio_percent")
        ax.plot(subset["valid_ratio_percent"], subset["gpu_ms_median"], marker="o", linewidth=2.0, label=variant)
    ax.set_title("Experiment 32: GPU Time by Valid Ratio")
    ax.set_xlabel("valid ratio (%)")
    ax.set_ylabel("GPU ms (median)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "subgroup_compaction_gpu_ms.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "subgroup_compaction_gpu_ms.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    subgroup = relative[relative["variant"].str.startswith("subgroup_ballot")].sort_values("valid_ratio_percent")
    ax.plot(subgroup["valid_ratio_percent"], subgroup["speedup_vs_shared_atomic_block"], marker="o", linewidth=2.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 32: subgroup_ballot Speedup vs shared_atomic_block")
    ax.set_xlabel("valid ratio (%)")
    ax.set_ylabel("speedup factor")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "subgroup_compaction_speedup.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "subgroup_compaction_speedup.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in stability["variant"].unique():
        subset = stability[stability["variant"] == variant].sort_values("valid_ratio_percent")
        ax.plot(subset["valid_ratio_percent"], subset["p95_to_median_gpu_ms"], marker="o", linewidth=2.0, label=variant)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 32: Stability by Valid Ratio")
    ax.set_xlabel("valid ratio (%)")
    ax.set_ylabel("p95 / median GPU ms")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "subgroup_compaction_stability.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "subgroup_compaction_stability.png")
    plt.close(fig)

    print(f"[ok] Wrote Experiment 32 charts to {CHARTS_DIR}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
