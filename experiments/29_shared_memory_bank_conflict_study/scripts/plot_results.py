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

EXPERIMENT_ID = "29_shared_memory_bank_conflict_study"

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
    summary = _load("shared_memory_bank_conflict_study_summary.csv")
    relative = _load("shared_memory_bank_conflict_study_relative.csv")
    stability = _load("shared_memory_bank_conflict_study_stability.csv")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary["shared_stride_elements"], summary["gpu_ms_median"], marker="o", linewidth=2.0)
    ax.set_title("Experiment 29: GPU Time by Shared-Memory Stride")
    ax.set_xlabel("shared stride (elements)")
    ax.set_ylabel("GPU ms (median)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "bank_conflict_gpu_ms.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "bank_conflict_gpu_ms.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(relative["shared_stride_elements"], relative["slowdown_vs_stride_1"], marker="o", linewidth=2.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 29: Slowdown vs Stride-1 Baseline")
    ax.set_xlabel("shared stride (elements)")
    ax.set_ylabel("slowdown factor")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "bank_conflict_slowdown.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "bank_conflict_slowdown.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stability["shared_stride_elements"], stability["p95_to_median_gpu_ms"], marker="o", linewidth=2.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 29: Stability by Shared-Memory Stride")
    ax.set_xlabel("shared stride (elements)")
    ax.set_ylabel("p95 / median GPU ms")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "bank_conflict_stability.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "bank_conflict_stability.png")
    plt.close(fig)

    print(f"[ok] Wrote Experiment 29 charts to {CHARTS_DIR}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
