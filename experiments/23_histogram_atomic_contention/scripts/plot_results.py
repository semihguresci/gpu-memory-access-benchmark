#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "histogram_atomic_contention_summary.csv"
RELATIVE_CSV = TABLES_DIR / "histogram_atomic_contention_relative.csv"
STABILITY_CSV = TABLES_DIR / "histogram_atomic_contention_stability.csv"

DISTRIBUTION_ORDER = {
    "uniform": 0,
    "hot_bin_90": 1,
    "mixed_hotset_75": 2,
}

IMPLEMENTATION_STYLE = {
    "global_atomics": {"label": "global_atomics", "color": "tab:blue"},
    "privatized_shared": {"label": "privatized_shared", "color": "tab:orange"},
}


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")
    return frame


def _sort_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    sorted_frame = frame.copy()
    sorted_frame["distribution_order"] = sorted_frame["distribution"].map(
        lambda value: DISTRIBUTION_ORDER.get(str(value), len(DISTRIBUTION_ORDER))
    )
    return sorted_frame.sort_values(["distribution_order"]).reset_index(drop=True)


def _plot_grouped_bars(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> None:
    plot_frame = _sort_distribution(frame)
    distributions = [str(value) for value in plot_frame["distribution"].unique()]
    x_positions = list(range(len(distributions)))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for implementation_index, (implementation, style) in enumerate(IMPLEMENTATION_STYLE.items()):
        subset = plot_frame[plot_frame["implementation"] == implementation]
        if subset.empty:
            continue
        offsets = [position + ((implementation_index - 0.5) * bar_width) for position in x_positions]
        ax.bar(offsets, subset[y_column].astype(float).to_list(), width=bar_width, color=style["color"],
               label=style["label"])

    ax.set_title(title)
    ax.set_xlabel("distribution")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions, distributions)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_relative(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> None:
    plot_frame = _sort_distribution(frame)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_frame["distribution"], plot_frame[y_column], marker="o", linewidth=2.0, color="tab:red")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("distribution")
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

    _plot_grouped_bars(
        summary,
        "gpu_ms_median",
        CHARTS_DIR / "histogram_atomic_contention_median_gpu_ms.png",
        "Experiment 23: Median GPU Time by Distribution",
        "GPU ms (median)",
    )
    _plot_grouped_bars(
        summary,
        "gbps_median",
        CHARTS_DIR / "histogram_atomic_contention_estimated_gbps.png",
        "Experiment 23: Estimated Global Traffic GB/s by Distribution",
        "Estimated GB/s",
    )
    _plot_relative(
        relative,
        "privatized_speedup_vs_global",
        CHARTS_DIR / "histogram_atomic_contention_speedup_vs_global.png",
        "Experiment 23: privatized_shared Speedup vs global_atomics",
        "Speedup factor",
    )
    _plot_grouped_bars(
        stability,
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "histogram_atomic_contention_stability_ratio.png",
        "Experiment 23: GPU Time Stability by Distribution",
        "p95 / median GPU ms",
    )

    print(f"[ok] Wrote Experiment 23 charts to {CHARTS_DIR.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
