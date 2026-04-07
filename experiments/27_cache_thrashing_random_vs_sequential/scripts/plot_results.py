#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

SUMMARY_CSV = TABLES_DIR / "cache_thrashing_random_vs_sequential_summary.csv"
RELATIVE_CSV = TABLES_DIR / "cache_thrashing_random_vs_sequential_relative.csv"
STABILITY_CSV = TABLES_DIR / "cache_thrashing_random_vs_sequential_stability.csv"

VARIANT_STYLE = {
    "sequential": {"label": "sequential", "color": "tab:blue", "linestyle": "-"},
    "block_shuffled": {"label": "block_shuffled", "color": "tab:orange", "linestyle": "--"},
    "random": {"label": "random", "color": "tab:red", "linestyle": "-."},
}


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")
    return frame


def _with_working_set_mib(frame: pd.DataFrame) -> pd.DataFrame:
    plot_frame = frame.copy()
    plot_frame["working_set_mib"] = plot_frame["working_set_bytes"].astype(float) / (1024.0 * 1024.0)
    return plot_frame.sort_values(["logical_elements", "variant"]).reset_index(drop=True)


def _plot_lines(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> None:
    plot_frame = _with_working_set_mib(frame)
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant, style in VARIANT_STYLE.items():
        subset = plot_frame[plot_frame["variant"] == variant]
        if subset.empty:
            continue
        ax.plot(
            subset["working_set_mib"],
            subset[y_column].astype(float),
            marker="o",
            linewidth=2.0,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )

    ax.set_title(title)
    ax.set_xlabel("working set (MiB per buffer)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_relative(frame: pd.DataFrame, y_column: str, output_path: Path, title: str, ylabel: str) -> None:
    plot_frame = frame.copy()
    plot_frame["working_set_mib"] = plot_frame["logical_elements"].astype(float) * 4.0 / (1024.0 * 1024.0)
    plot_frame = plot_frame.sort_values(["logical_elements", "variant"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, style in VARIANT_STYLE.items():
        if variant == "sequential":
            continue
        subset = plot_frame[plot_frame["variant"] == variant]
        if subset.empty:
            continue
        ax.plot(
            subset["working_set_mib"],
            subset[y_column].astype(float),
            marker="o",
            linewidth=2.0,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("working set (MiB per buffer)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    summary = _load_table(SUMMARY_CSV)
    relative = _load_table(RELATIVE_CSV)
    stability = _load_table(STABILITY_CSV)

    _plot_lines(
        summary,
        "gpu_ms_median",
        CHARTS_DIR / "cache_thrashing_random_vs_sequential_median_gpu_ms.png",
        "Experiment 27: Median GPU Time by Access Pattern",
        "GPU ms (median)",
    )
    _plot_relative(
        relative,
        "slowdown_vs_sequential",
        CHARTS_DIR / "cache_thrashing_random_vs_sequential_slowdown_vs_sequential.png",
        "Experiment 27: Slowdown vs Sequential",
        "Slowdown factor",
    )
    _plot_lines(
        summary,
        "gbps_median",
        CHARTS_DIR / "cache_thrashing_random_vs_sequential_estimated_gbps.png",
        "Experiment 27: Estimated GB/s by Access Pattern",
        "Estimated GB/s",
    )
    _plot_lines(
        stability,
        "p95_to_median_gpu_ms",
        CHARTS_DIR / "cache_thrashing_random_vs_sequential_stability_ratio.png",
        "Experiment 27: GPU Time Stability",
        "p95 / median GPU ms",
    )

    print(f"[ok] Wrote Experiment 27 charts to {CHARTS_DIR.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
