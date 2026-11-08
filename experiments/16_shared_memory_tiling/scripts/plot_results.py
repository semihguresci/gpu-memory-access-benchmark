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

EXPERIMENT_ID = "16_shared_memory_tiling"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "shared_memory_tiling_summary.csv"
RELATIVE_CSV = TABLES_DIR / "shared_memory_tiling_relative.csv"
STABILITY_CSV = TABLES_DIR / "shared_memory_tiling_stability.csv"

IMPLEMENTATION_STYLE = {
    "direct_global": {"label": "direct_global", "color": "tab:blue", "marker": "o"},
    "shared_tiled": {"label": "shared_tiled", "color": "tab:orange", "marker": "s"},
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


def _plot_metric_by_radius(
    frame: pd.DataFrame,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    *,
    yscale: str | None = None,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for implementation, style in IMPLEMENTATION_STYLE.items():
        subset = frame[frame["implementation"] == implementation].sort_values("reuse_radius")
        if subset.empty:
            continue
        ax.plot(
            subset["reuse_radius"],
            subset[y_column],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            label=style["label"],
        )

    ax.set_title(title)
    ax.set_xlabel("reuse radius")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(frame["reuse_radius"].unique()))
    ax.grid(True, alpha=0.3)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_relative(
    frame: pd.DataFrame,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    *,
    baseline: float | None = None,
) -> Path | None:
    plot_frame = frame.sort_values("reuse_radius")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_frame["reuse_radius"], plot_frame[y_column], marker="o", linewidth=2.0, color="tab:red")
    ax.set_title(title)
    ax.set_xlabel("reuse radius")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(plot_frame["reuse_radius"].unique()))
    ax.grid(True, alpha=0.3)
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

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

    generated_charts.append(_plot_metric_by_radius(
        summary,
        "gpu_ms_median",
        CHARTS_DIR / "shared_memory_tiling_median_gpu_ms.png",
        "Experiment 16: Median GPU Time by Reuse Radius",
        "GPU ms (median)",
    ))
    generated_charts.append(_plot_metric_by_radius(
        summary,
        "gbps_median",
        CHARTS_DIR / "shared_memory_tiling_estimated_gbps.png",
        "Experiment 16: Estimated Global Traffic GB/s by Reuse Radius",
        "Estimated GB/s",
    ))
    generated_charts.append(_plot_relative(
        relative,
        "tiled_speedup_vs_direct",
        CHARTS_DIR / "shared_memory_tiling_speedup_vs_direct.png",
        "Experiment 16: shared_tiled Speedup vs direct_global",
        "Speedup factor",
        baseline=1.0,
    ))
    generated_charts.append(_plot_metric_by_radius(
        stability,
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "shared_memory_tiling_stability_ratio.png",
        "Experiment 16: GPU Time Stability by Reuse Radius",
        "p95 / median GPU ms",
    ))

    print(f"[ok] Wrote Experiment 16 charts to {CHARTS_DIR.relative_to(ROOT)}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
