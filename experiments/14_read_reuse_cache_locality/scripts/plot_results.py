#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "read_reuse_cache_locality_summary.csv"
RELATIVE_CSV = TABLES_DIR / "read_reuse_cache_locality_relative.csv"
STABILITY_CSV = TABLES_DIR / "read_reuse_cache_locality_stability.csv"

VARIANT_ORDER = {
    "reuse_distance_1": 0,
    "reuse_distance_32": 1,
    "reuse_distance_256": 2,
    "reuse_distance_4096": 3,
    "reuse_distance_full_span": 4,
}


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")

    return frame


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sorted_frame = frame.copy()
    if "variant" in sorted_frame.columns:
        sorted_frame["variant_order"] = sorted_frame["variant"].map(lambda value: VARIANT_ORDER.get(str(value), 99))
        sort_columns = ["variant_order"]
        if "logical_elements" in sorted_frame.columns:
            sort_columns.append("logical_elements")
        sorted_frame = sorted_frame.sort_values(sort_columns).reset_index(drop=True)
    return sorted_frame


def _plot_bars(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    *,
    yscale: str | None = None,
    baseline: float | None = None,
    color: str = "tab:blue",
) -> None:
    plot_frame = _sort_frame(frame)
    x_values = plot_frame[x_column].astype(str).to_list()
    y_values = plot_frame[y_column].astype(float).to_list()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_values, y_values, color=color, width=0.7)
    ax.set_title(title)
    ax.set_xlabel("variant")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    if yscale is not None:
        ax.set_yscale(yscale)
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    summary = _load_table(SUMMARY_CSV)
    relative = _load_table(RELATIVE_CSV)
    stability = _load_table(STABILITY_CSV)

    _plot_bars(
        summary,
        "variant",
        "gpu_ms_median",
        CHARTS_DIR / "read_reuse_cache_locality_median_gpu_ms.png",
        "Experiment 14: Median GPU Time by Reuse Distance",
        "GPU ms (median)",
        yscale="log",
        color="tab:blue",
    )
    _plot_bars(
        summary,
        "variant",
        "gbps_median",
        CHARTS_DIR / "read_reuse_cache_locality_median_gbps.png",
        "Experiment 14: Logical Traffic Proxy by Reuse Distance",
        "GB/s proxy (median)",
        yscale="log",
        color="tab:green",
    )
    _plot_bars(
        relative,
        "variant",
        "speedup_vs_full_span",
        CHARTS_DIR / "read_reuse_cache_locality_speedup_vs_full_span.png",
        "Experiment 14: Speedup vs reuse_distance_full_span",
        "Speedup factor",
        baseline=1.0,
        color="tab:red",
    )
    _plot_bars(
        stability,
        "variant",
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "read_reuse_cache_locality_stability_ratio.png",
        "Experiment 14: GPU Time Stability by Reuse Distance",
        "p95 / median GPU ms",
        baseline=1.0,
        color="tab:orange",
    )

    print(f"[ok] Wrote Experiment 14 charts to {CHARTS_DIR.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
