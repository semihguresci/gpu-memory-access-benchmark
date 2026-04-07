#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "stream_compaction_summary.csv"
RELATIVE_CSV = TABLES_DIR / "stream_compaction_relative.csv"
STABILITY_CSV = TABLES_DIR / "stream_compaction_stability.csv"

IMPLEMENTATION_STYLE = {
    "global_atomic_append": {"label": "global_atomic_append", "color": "tab:blue"},
    "three_stage": {"label": "three_stage", "color": "tab:orange"},
}


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")
    return frame


def _plot_grouped_bars(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> None:
    ratios = sorted(frame["valid_ratio_percent"].astype(int).unique().tolist())
    x_positions = list(range(len(ratios)))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for implementation_index, (implementation, style) in enumerate(IMPLEMENTATION_STYLE.items()):
        subset = frame[frame["implementation"] == implementation].sort_values(["valid_ratio_percent"])
        if subset.empty:
            continue
        offsets = [position + ((implementation_index - 0.5) * bar_width) for position in x_positions]
        ax.bar(offsets, subset[y_column].astype(float).to_list(), width=bar_width, color=style["color"],
               label=style["label"])

    ax.set_title(title)
    ax.set_xlabel("valid ratio (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions, [str(ratio) for ratio in ratios])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_relative(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> None:
    plot_frame = frame.sort_values(["valid_ratio_percent"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_frame["valid_ratio_percent"], plot_frame[y_column], marker="o", linewidth=2.0, color="tab:red")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("valid ratio (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(plot_frame["valid_ratio_percent"].astype(int).tolist())
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
        CHARTS_DIR / "stream_compaction_median_gpu_ms.png",
        "Experiment 24: Median GPU Time by Valid Ratio",
        "GPU ms (median)",
    )
    _plot_grouped_bars(
        summary,
        "gbps_median",
        CHARTS_DIR / "stream_compaction_effective_gbps.png",
        "Experiment 24: Effective GB/s by Valid Ratio",
        "Effective GB/s",
    )
    _plot_relative(
        relative,
        "three_stage_speedup_vs_atomic",
        CHARTS_DIR / "stream_compaction_speedup_vs_atomic.png",
        "Experiment 24: three_stage Speedup vs global_atomic_append",
        "Speedup factor",
    )
    _plot_grouped_bars(
        stability,
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "stream_compaction_stability_ratio.png",
        "Experiment 24: GPU Time Stability by Valid Ratio",
        "p95 / median GPU ms",
    )

    print(f"[ok] Wrote Experiment 24 charts to {CHARTS_DIR.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
