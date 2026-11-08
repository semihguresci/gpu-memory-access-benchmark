#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analysis_integrity import (
    validate_analysis_provenance,
    validate_derived_table,
    validate_summary_table,
    write_plot_provenance,
)

EXPERIMENT_ID = "48_buffer_vs_image_representation"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"

REPRESENTATION_COLORS = {
    "row_major_buffer": "tab:blue",
    "column_major_buffer": "tab:orange",
    "storage_image": "tab:green",
    "sampled_image": "tab:red",
}
MODE_STYLES = {"neighborhood_clamped": "-", "random_scatter": "--"}


def _load(path: str) -> pd.DataFrame:
    validate_derived_table(TABLES_DIR / path)
    frame = pd.read_csv(TABLES_DIR / path)
    validate_summary_table(frame, TABLES_DIR / path)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")
    return frame


def main() -> None:
    generated_charts: list[Path | None] = []
    validate_analysis_provenance(TABLES_DIR, EXPERIMENT_ID)
    summary = _load("buffer_vs_image_representation_summary.csv")
    effects = _load("buffer_vs_image_representation_representation_effects.csv")
    stability = _load("buffer_vs_image_representation_stability.csv")
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in summary["variant"].unique():
        representation = str(variant).rsplit("_", 1)[0]
        mode = str(variant)[len(representation) + 1:]
        subset = summary[summary["variant"] == variant].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["gpu_ms_median"], marker="o", linewidth=1.8,
                color=REPRESENTATION_COLORS.get(representation, None),
                linestyle=MODE_STYLES.get(mode, "-"), label=variant)
    ax.set_title("Experiment 48: GPU Time by Element Count")
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("GPU ms (median)")
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "buffer_vs_image_representation_ms_vs_size.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "buffer_vs_image_representation_ms_vs_size.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for representation, color in REPRESENTATION_COLORS.items():
        subset = effects[
            (effects["representation"] == representation) & (effects["access_mode"] == "neighborhood_clamped")
        ].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["gpu_time_ratio_vs_row_buffer"], marker="o", linewidth=2.0,
                color=color, label=f"{representation} (neighborhood clamped)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 48: Representation Time Ratio vs Row Buffer")
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("median GPU time ratio")
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "buffer_vs_image_representation_ratios_neighborhood.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "buffer_vs_image_representation_ratios_neighborhood.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for representation, color in REPRESENTATION_COLORS.items():
        subset = effects[
            (effects["representation"] == representation) & (effects["access_mode"] == "random_scatter")
        ].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["gpu_time_ratio_vs_row_buffer"], marker="o", linewidth=2.0,
                color=color, label=f"{representation} (random scatter)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 48: Representation Time Ratio vs Row Buffer (Random Scatter)")
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("median GPU time ratio")
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "buffer_vs_image_representation_ratios_random.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "buffer_vs_image_representation_ratios_random.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in stability["variant"].unique():
        subset = stability[stability["variant"] == variant].sort_values("problem_size")
        ax.plot(subset["problem_size"], subset["p95_to_median_gpu_ms"], marker="o", linewidth=1.4, label=variant)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Experiment 48: Stability by Element Count")
    ax.set_xlabel("problem size (elements)")
    ax.set_ylabel("p95 / median GPU ms")
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "buffer_vs_image_representation_stability.png", dpi=150)
    generated_charts.append(CHARTS_DIR / "buffer_vs_image_representation_stability.png")
    plt.close(fig)

    print(f"[ok] Wrote Experiment 48 charts to {CHARTS_DIR}.")
    write_plot_provenance(TABLES_DIR, EXPERIMENT_ID, generated_charts, plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
