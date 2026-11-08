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

EXPERIMENT_ID = "21_parallel_reduction"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "parallel_reduction_summary.csv"
RELATIVE_CSV = TABLES_DIR / "parallel_reduction_relative.csv"
STABILITY_CSV = TABLES_DIR / "parallel_reduction_stability.csv"

VARIANT_STYLE = {
    "global_atomic": {"label": "global_atomic", "color": "tab:red", "marker": "o"},
    "shared_tree": {"label": "shared_tree", "color": "tab:green", "marker": "s"},
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


def _plot_metric_by_size(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant, style in VARIANT_STYLE.items():
        subset = frame[frame["variant"] == variant].sort_values("input_mib")
        if subset.empty:
            continue
        ax.plot(
            subset["input_mib"],
            subset[y_column],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            label=style["label"],
        )

    ax.set_title(title)
    ax.set_xlabel("input size (MiB)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_relative(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = frame.sort_values("input_mib")
    ax.plot(subset["input_mib"], subset[y_column], marker="o", linewidth=2.0, color="tab:blue")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("input size (MiB)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
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

    generated_charts.append(_plot_metric_by_size(
        summary,
        "gpu_ms_median",
        CHARTS_DIR / "parallel_reduction_median_gpu_ms.png",
        "Experiment 21: Median GPU Time by Input Size",
        "GPU ms (median)",
    ))
    generated_charts.append(_plot_metric_by_size(
        summary,
        "gbps_median",
        CHARTS_DIR / "parallel_reduction_gbps.png",
        "Experiment 21: Logical Bytes Touched per Second",
        "GB/s (median)",
    ))
    generated_charts.append(_plot_relative(
        relative,
        "shared_tree_speedup_vs_global_atomic",
        CHARTS_DIR / "parallel_reduction_speedup_vs_global_atomic.png",
        "Experiment 21: shared_tree Speedup vs global_atomic",
        "Speedup factor",
    ))
    generated_charts.append(_plot_metric_by_size(
        stability,
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "parallel_reduction_stability_ratio.png",
        "Experiment 21: GPU Time Stability by Input Size",
        "p95 / median GPU ms",
    ))

    print(f"[ok] Wrote Experiment 21 charts to {CHARTS_DIR.relative_to(ROOT)}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
