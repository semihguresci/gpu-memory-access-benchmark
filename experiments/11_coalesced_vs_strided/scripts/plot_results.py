#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analysis_integrity import (
    validate_analysis_provenance,
    validate_derived_table,
    validate_summary_table,
    write_plot_provenance,
)

EXPERIMENT_ID = "11_coalesced_vs_strided"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "coalesced_vs_strided_summary.csv"
RELATIVE_CSV = TABLES_DIR / "coalesced_vs_strided_relative.csv"
STABILITY_CSV = TABLES_DIR / "coalesced_vs_strided_stability.csv"


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")

    validate_derived_table(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")

    validate_summary_table(frame, path)
    return frame


def _format_stride_ticks(values: np.ndarray) -> list[str]:
    return [str(int(value)) for value in values]


def _prepare_axes(ax: plt.Axes, title: str, ylabel: str, yscale: str | None = None, baseline: float | None = None) -> None:
    ax.set_title(title)
    ax.set_xlabel("stride")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xscale("log", base=2)
    if yscale is not None:
        ax.set_yscale(yscale)
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.0, alpha=0.7)


def _plot_series(frame: pd.DataFrame, x_column: str, y_column: str, output_path: Path, title: str, ylabel: str, *, yscale: str | None = None, baseline: float | None = None, color: str = "tab:blue") -> Path | None:
    plot_frame = frame.sort_values(x_column).reset_index(drop=True)
    x_values = plot_frame[x_column].to_numpy(dtype=float)
    y_values = plot_frame[y_column].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, y_values, marker="o", linewidth=2.5, markersize=6, color=color)
    _prepare_axes(ax, title=title, ylabel=ylabel, yscale=yscale, baseline=baseline)
    ax.set_xticks(x_values)
    ax.set_xticklabels(_format_stride_ticks(x_values))
    ax.set_xlim(x_values.min() * 0.9, x_values.max() * 1.1)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    generated_charts: list[Path | None] = []
    validate_analysis_provenance(TABLES_DIR, "11_coalesced_vs_strided")
    summary = _load_table(SUMMARY_CSV)
    relative = _load_table(RELATIVE_CSV)
    stability = _load_table(STABILITY_CSV)

    generated_charts.append(_plot_series(
        summary,
        "stride",
        "gpu_ms_median",
        CHARTS_DIR / "coalesced_vs_strided_median_gpu_ms.png",
        "Experiment 11: Median GPU Time by Stride",
        "GPU ms (median, log scale)",
        yscale="log",
        color="tab:blue",
    ))
    generated_charts.append(_plot_series(
        summary,
        "stride",
        "gbps_median",
        CHARTS_DIR / "coalesced_vs_strided_median_gbps.png",
        "Experiment 11: Median Bandwidth by Stride",
        "GB/s (median, log scale)",
        yscale="log",
        color="tab:green",
    ))
    generated_charts.append(_plot_series(
        relative,
        "stride",
        "slowdown_vs_stride_1",
        CHARTS_DIR / "coalesced_vs_strided_slowdown_vs_stride_1.png",
        "Experiment 11: Slowdown vs stride_1",
        "Slowdown factor (higher is slower, log scale)",
        yscale="log",
        baseline=1.0,
        color="tab:red",
    ))
    generated_charts.append(_plot_series(
        stability,
        "stride",
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "coalesced_vs_strided_stability_ratio.png",
        "Experiment 11: GPU Time Stability by Stride",
        "p95 / median GPU ms",
        yscale=None,
        baseline=1.0,
        color="tab:orange",
    ))

    print(f"[ok] Wrote Experiment 11 charts to {CHARTS_DIR.relative_to(ROOT)}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
