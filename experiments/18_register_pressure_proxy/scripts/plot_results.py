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

EXPERIMENT_ID = "18_register_pressure_proxy"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "register_pressure_proxy_summary.csv"
RELATIVE_CSV = TABLES_DIR / "register_pressure_proxy_relative.csv"
STABILITY_CSV = TABLES_DIR / "register_pressure_proxy_stability.csv"


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")

    validate_derived_table(path)
    frame = pd.read_csv(path)
    validate_summary_table(frame, path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")

    return frame


def _plot_metric(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> Path | None:
    plot_frame = frame.sort_values("temporary_count").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(plot_frame["temporary_count"].astype(str), plot_frame[y_column], color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("temporary count")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_relative(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> Path | None:
    plot_frame = frame.sort_values("temporary_count").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_frame["temporary_count"], plot_frame[y_column], marker="o", linewidth=2.0, color="tab:red")
    ax.set_title(title)
    ax.set_xlabel("temporary count")
    ax.set_ylabel(ylabel)
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

    generated_charts.append(_plot_metric(
        summary,
        "gpu_ms_median",
        CHARTS_DIR / "register_pressure_proxy_median_gpu_ms.png",
        "Experiment 18: Median GPU Time by Temporary Count",
        "GPU ms (median)",
    ))
    generated_charts.append(_plot_relative(
        relative,
        "speedup_vs_baseline",
        CHARTS_DIR / "register_pressure_proxy_speedup_vs_baseline.png",
        "Experiment 18: Speedup vs Lightest Variant",
        "Speedup factor",
    ))
    generated_charts.append(_plot_relative(
        stability,
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "register_pressure_proxy_stability_ratio.png",
        "Experiment 18: GPU Time Stability by Temporary Count",
        "p95 / median GPU ms",
    ))

    print(f"[ok] Wrote Experiment 18 charts to {CHARTS_DIR.relative_to(ROOT)}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
