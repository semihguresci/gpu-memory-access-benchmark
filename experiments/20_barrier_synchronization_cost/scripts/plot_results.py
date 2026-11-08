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

EXPERIMENT_ID = "20_barrier_synchronization_cost"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "barrier_synchronization_cost_summary.csv"
RELATIVE_CSV = TABLES_DIR / "barrier_synchronization_cost_relative.csv"
STABILITY_CSV = TABLES_DIR / "barrier_synchronization_cost_stability.csv"

PLACEMENT_STYLE = {
    "flat_loop": {"label": "flat_loop", "color": "tab:blue", "marker": "o"},
    "tiled_regions": {"label": "tiled_regions", "color": "tab:orange", "marker": "s"},
}


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")

    validate_derived_table(path)
    frame = pd.read_csv(path)
    validate_summary_table(frame, path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")

    return frame


def _plot_summary_metric(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for placement, style in PLACEMENT_STYLE.items():
        subset = frame[frame["placement"] == placement].sort_values("barrier_interval_phases")
        if subset.empty:
            continue
        ax.plot(
            subset["barrier_interval_phases"],
            subset[y_column],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            label=style["label"],
        )

    ax.set_title(title)
    ax.set_xlabel("barrier interval (phases)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(frame["barrier_interval_phases"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_relative(frame: pd.DataFrame, output_path: Path, title: str, ylabel: str) -> Path | None:
    plot_frame = frame.sort_values("barrier_interval_phases")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_frame["barrier_interval_phases"], plot_frame["tiled_speedup_vs_flat"], marker="o", linewidth=2.0)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("barrier interval (phases)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(plot_frame["barrier_interval_phases"].unique()))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    generated_charts: list[Path | None] = []
    validate_analysis_provenance(TABLES_DIR, EXPERIMENT_ID)
    summary = _load_table(SUMMARY_CSV)
    relative = _load_table(RELATIVE_CSV)
    stability = _load_table(STABILITY_CSV)

    generated_charts.append(_plot_summary_metric(
        summary,
        "gpu_ms_median",
        CHARTS_DIR / "barrier_synchronization_cost_median_gpu_ms.png",
        "Experiment 20: Median GPU Time by Barrier Interval",
        "GPU ms (median)",
    ))
    generated_charts.append(_plot_summary_metric(
        summary,
        "gbps_median",
        CHARTS_DIR / "barrier_synchronization_cost_estimated_gbps.png",
        "Experiment 20: Estimated Logical GB/s by Barrier Interval",
        "Estimated GB/s",
    ))
    generated_charts.append(_plot_relative(
        relative,
        CHARTS_DIR / "barrier_synchronization_cost_speedup_vs_flat.png",
        "Experiment 20: tiled_regions Speedup vs flat_loop",
        "Speedup factor",
    ))
    generated_charts.append(_plot_summary_metric(
        stability,
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "barrier_synchronization_cost_stability_ratio.png",
        "Experiment 20: GPU Time Stability by Barrier Interval",
        "p95 / median GPU ms",
    ))

    print(f"[ok] Wrote Experiment 20 charts to {CHARTS_DIR.relative_to(ROOT)}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
