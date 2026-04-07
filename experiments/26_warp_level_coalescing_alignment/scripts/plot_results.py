#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "warp_level_coalescing_alignment_summary.csv"
RELATIVE_CSV = TABLES_DIR / "warp_level_coalescing_alignment_relative.csv"
STABILITY_CSV = TABLES_DIR / "warp_level_coalescing_alignment_stability.csv"


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")

    return frame


def _plot_series(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    *,
    baseline: float | None = None,
    color: str = "tab:blue",
) -> None:
    plot_frame = frame.sort_values(x_column).reset_index(drop=True)
    x_values = plot_frame[x_column].to_numpy(dtype=float)
    y_values = plot_frame[y_column].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, y_values, marker="o", linewidth=2.5, markersize=6, color=color)
    ax.set_title(title)
    ax.set_xlabel("alignment offset (bytes)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(x_values)
    ax.set_xlim(x_values.min() - 1, x_values.max() + 1)
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

    aligned_summary = summary[summary["alignment_offset_bytes"] == 0].sort_values("problem_size")
    if aligned_summary.empty:
        raise ValueError("Missing aligned baseline row in summary table.")

    _plot_series(
        summary,
        "alignment_offset_bytes",
        "gpu_ms_median",
        CHARTS_DIR / "warp_level_coalescing_alignment_median_gpu_ms.png",
        "Experiment 26: Median GPU Time by Alignment Offset",
        "GPU ms (median)",
        color="tab:blue",
    )
    _plot_series(
        summary,
        "alignment_offset_bytes",
        "gbps_median",
        CHARTS_DIR / "warp_level_coalescing_alignment_median_gbps.png",
        "Experiment 26: Median Effective Bandwidth by Alignment Offset",
        "GB/s (median)",
        color="tab:green",
    )
    _plot_series(
        relative,
        "alignment_offset_bytes",
        "slowdown_vs_aligned",
        CHARTS_DIR / "warp_level_coalescing_alignment_slowdown_vs_aligned.png",
        "Experiment 26: Slowdown vs Aligned Baseline",
        "Slowdown factor",
        baseline=1.0,
        color="tab:red",
    )
    _plot_series(
        stability,
        "alignment_offset_bytes",
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "warp_level_coalescing_alignment_stability_ratio.png",
        "Experiment 26: GPU Time Stability by Alignment Offset",
        "p95 / median GPU ms",
        baseline=1.0,
        color="tab:orange",
    )

    resolved_charts = CHARTS_DIR.resolve()
    try:
        charts_label = resolved_charts.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        charts_label = resolved_charts.as_posix()
    print(f"[ok] Wrote Experiment 26 charts to {charts_label}.")


if __name__ == "__main__":
    main()
