#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")

    return frame


def _plot_metric(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> None:
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


def _plot_relative(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> None:
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


def main() -> None:
    summary = _load_table(SUMMARY_CSV)
    relative = _load_table(RELATIVE_CSV)
    stability = _load_table(STABILITY_CSV)

    _plot_metric(
        summary,
        "gpu_ms_median",
        CHARTS_DIR / "register_pressure_proxy_median_gpu_ms.png",
        "Experiment 18: Median GPU Time by Temporary Count",
        "GPU ms (median)",
    )
    _plot_relative(
        relative,
        "speedup_vs_baseline",
        CHARTS_DIR / "register_pressure_proxy_speedup_vs_baseline.png",
        "Experiment 18: Speedup vs Lightest Variant",
        "Speedup factor",
    )
    _plot_relative(
        stability,
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "register_pressure_proxy_stability_ratio.png",
        "Experiment 18: GPU Time Stability by Temporary Count",
        "p95 / median GPU ms",
    )

    print(f"[ok] Wrote Experiment 18 charts to {CHARTS_DIR.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
