#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "branch_divergence_summary.csv"
RELATIVE_CSV = TABLES_DIR / "branch_divergence_relative.csv"
STABILITY_CSV = TABLES_DIR / "branch_divergence_stability.csv"


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")

    return frame


def _select_plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    max_problem_size = frame["problem_size"].max()
    return frame[frame["problem_size"] == max_problem_size].sort_values("pattern_order").reset_index(drop=True)


def _plot_series(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str, *, baseline: float | None = None, color: str = "tab:blue") -> None:
    plot_frame = _select_plot_frame(frame)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_frame["pattern_order"], plot_frame[y_column], marker="o", linewidth=2.5, markersize=6, color=color)
    ax.set_title(title)
    ax.set_xlabel("branch pattern")
    ax.set_ylabel(ylabel)
    ax.set_xticks(plot_frame["pattern_order"])
    ax.set_xticklabels(plot_frame["pattern_mode"], rotation=20, ha="right")
    ax.grid(True, alpha=0.3)
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    summary = _load_table(SUMMARY_CSV)
    relative = _load_table(RELATIVE_CSV)
    stability = _load_table(STABILITY_CSV)

    _plot_series(
        summary,
        "gpu_ms_median",
        CHARTS_DIR / "branch_divergence_median_gpu_ms.png",
        "Experiment 19: Median GPU Time by Branch Pattern",
        "GPU ms (median)",
    )
    _plot_series(
        summary,
        "gbps_median",
        CHARTS_DIR / "branch_divergence_median_gbps.png",
        "Experiment 19: Median Bandwidth by Branch Pattern",
        "GB/s (median)",
        color="tab:green",
    )
    _plot_series(
        relative,
        "slowdown_vs_uniform_true",
        CHARTS_DIR / "branch_divergence_slowdown_vs_uniform_true.png",
        "Experiment 19: Slowdown vs uniform_true",
        "Slowdown factor",
        baseline=1.0,
        color="tab:red",
    )
    _plot_series(
        stability,
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "branch_divergence_stability_ratio.png",
        "Experiment 19: GPU Time Stability by Branch Pattern",
        "p95 / median GPU ms",
        baseline=1.0,
        color="tab:orange",
    )

    print(f"[ok] Wrote Experiment 19 charts to {CHARTS_DIR.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
