#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "tile_size_sweep_summary.csv"
RELATIVE_CSV = TABLES_DIR / "tile_size_sweep_relative.csv"
STABILITY_CSV = TABLES_DIR / "tile_size_sweep_stability.csv"

IMPLEMENTATION_STYLE = {
    "direct_global": {"label": "direct_global", "color": "tab:blue", "marker": "o"},
    "shared_tiled": {"label": "shared_tiled", "color": "tab:orange", "marker": "s"},
}


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")

    return frame


def _plot_metric_by_tile_size(
    frame: pd.DataFrame,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    *,
    yscale: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for implementation, style in IMPLEMENTATION_STYLE.items():
        subset = frame[frame["implementation"] == implementation].sort_values("tile_size")
        if subset.empty:
            continue
        ax.plot(
            subset["tile_size"],
            subset[y_column],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            label=style["label"],
        )

    ax.set_title(title)
    ax.set_xlabel("tile size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(frame["tile_size"].unique()))
    ax.grid(True, alpha=0.3)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_relative(
    frame: pd.DataFrame,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    *,
    baseline: float | None = None,
) -> None:
    plot_frame = frame.sort_values("tile_size")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_frame["tile_size"], plot_frame[y_column], marker="o", linewidth=2.0, color="tab:red")
    ax.set_title(title)
    ax.set_xlabel("tile size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(plot_frame["tile_size"].unique()))
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

    _plot_metric_by_tile_size(
        summary,
        "gpu_ms_median",
        CHARTS_DIR / "tile_size_sweep_median_gpu_ms.png",
        "Experiment 17: Median GPU Time by Tile Size",
        "GPU ms (median)",
    )
    _plot_metric_by_tile_size(
        summary,
        "gbps_median",
        CHARTS_DIR / "tile_size_sweep_estimated_gbps.png",
        "Experiment 17: Estimated Global Traffic GB/s by Tile Size",
        "Estimated GB/s",
    )
    _plot_relative(
        relative,
        "tiled_speedup_vs_direct",
        CHARTS_DIR / "tile_size_sweep_speedup_vs_direct.png",
        "Experiment 17: shared_tiled Speedup vs direct_global",
        "Speedup factor",
        baseline=1.0,
    )
    _plot_metric_by_tile_size(
        stability,
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "tile_size_sweep_stability_ratio.png",
        "Experiment 17: GPU Time Stability by Tile Size",
        "p95 / median GPU ms",
    )

    print(f"[ok] Wrote Experiment 17 charts to {CHARTS_DIR.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
