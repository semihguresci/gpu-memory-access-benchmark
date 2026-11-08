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

EXPERIMENT_ID = "22_prefix_sum_scan"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "prefix_sum_scan_summary.csv"
RELATIVE_CSV = TABLES_DIR / "prefix_sum_scan_relative.csv"
STABILITY_CSV = TABLES_DIR / "prefix_sum_scan_stability.csv"

VARIANT_STYLE = {
    "items_per_thread_1": {"label": "items/thread = 1", "color": "tab:blue", "marker": "o"},
    "items_per_thread_2": {"label": "items/thread = 2", "color": "tab:orange", "marker": "s"},
    "items_per_thread_4": {"label": "items/thread = 4", "color": "tab:green", "marker": "^"},
    "items_per_thread_8": {"label": "items/thread = 8", "color": "tab:red", "marker": "D"},
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


def _plot_curve(frame: pd.DataFrame, x_column: str, y_column: str, output_path: Path, title: str, ylabel: str) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, style in VARIANT_STYLE.items():
        subset = frame[frame["variant"] == variant].sort_values(x_column)
        if subset.empty:
            continue
        ax.plot(
            subset[x_column],
            subset[y_column],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            label=style["label"],
        )

    ax.set_title(title)
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_relative(frame: pd.DataFrame, output_path: Path, title: str) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, style in VARIANT_STYLE.items():
        subset = frame[frame["variant"] == variant].sort_values("size_bytes")
        if subset.empty:
            continue
        ax.plot(
            subset["problem_size"],
            subset["speedup_vs_baseline"],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            label=style["label"],
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("speedup vs items/thread = 1")
    ax.grid(True, alpha=0.3)
    ax.legend()

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

    generated_charts.append(_plot_curve(
        summary,
        "problem_size",
        "gpu_ms_median",
        CHARTS_DIR / "prefix_sum_scan_median_gpu_ms.png",
        "Experiment 22: Median GPU Time by Items per Thread",
        "GPU ms (median)",
    ))
    generated_charts.append(_plot_curve(
        summary,
        "problem_size",
        "gbps_median",
        CHARTS_DIR / "prefix_sum_scan_effective_gbps.png",
        "Experiment 22: Estimated Storage-Buffer Requests by Items per Thread",
        "Estimated source-request GB/s (median)",
    ))
    generated_charts.append(_plot_relative(
        relative,
        CHARTS_DIR / "prefix_sum_scan_speedup_vs_baseline.png",
        "Experiment 22: Speedup vs items/thread = 1",
    ))

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, style in VARIANT_STYLE.items():
        subset = stability[stability["variant"] == variant].sort_values("problem_size")
        if subset.empty:
            continue
        ax.plot(
            subset["problem_size"],
            subset["p95_to_median_gpu_ms"],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            label=style["label"],
        )
    ax.set_title("Experiment 22: GPU Time Stability")
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("p95 / median GPU ms")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / "prefix_sum_scan_stability_ratio.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "prefix_sum_scan_stability_ratio.png")
    plt.close(fig)

    print(f"[ok] Wrote Experiment 22 charts to {CHARTS_DIR.relative_to(ROOT)}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
