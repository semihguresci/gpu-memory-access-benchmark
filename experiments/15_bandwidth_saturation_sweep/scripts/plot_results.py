#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT / "results" / "tables" / "benchmark_results.json"
OUTPUT_PNG = ROOT / "results" / "charts" / "benchmark_summary.png"
EXPERIMENT_ID = "15_bandwidth_saturation_sweep"
BYTES_PER_ELEMENT = 4.0


def _format_size_ticks(values: np.ndarray) -> list[str]:
    return [str(int(value)) for value in values]


def _plot_bandwidth_saturation_summary() -> bool:
    if not INPUT_JSON.exists():
        return False

    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        return False

    df = pd.DataFrame(rows)
    df = df[df.get("experiment_id", "") == EXPERIMENT_ID].copy()
    if df.empty:
        return False

    for column in ["problem_size", "gpu_ms", "gbps"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["variant"] = df["variant"].fillna("unknown_variant").astype(str)
    df = df.dropna(subset=["problem_size", "gpu_ms", "gbps"])
    if df.empty:
        return False

    df["size_mib"] = (df["problem_size"] * BYTES_PER_ELEMENT) / (1024.0 * 1024.0)
    summary = (
        df.groupby(["variant", "size_mib"], as_index=False)
        .agg(gpu_ms_median=("gpu_ms", "median"), gbps_median=("gbps", "median"))
        .sort_values(["variant", "size_mib"])
    )

    variants = sorted(summary["variant"].unique().tolist())
    size_values = np.array(sorted(summary["size_mib"].unique().tolist()), dtype=float)
    if len(size_values) == 0:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for variant in variants:
        curve = summary[summary["variant"] == variant].sort_values("size_mib")
        axes[0].plot(curve["size_mib"], curve["gpu_ms_median"], marker="o", label=variant)
        axes[1].plot(curve["size_mib"], curve["gbps_median"], marker="o", label=variant)

    axes[0].set_title("Dispatch time (median)")
    axes[0].set_ylabel("gpu_ms")
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Effective bandwidth (median)")
    axes[1].set_ylabel("GB/s")
    axes[1].set_xscale("log", base=2)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("size (MiB)")
        ax.set_xticks(size_values)
        ax.set_xticklabels(_format_size_ticks(size_values), rotation=30)

    axes[1].legend(title="variant")
    fig.suptitle("Experiment 15 Bandwidth Saturation Sweep", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    if not _plot_bandwidth_saturation_summary():
        raise FileNotFoundError(
            "No Experiment 15 rows found in benchmark_results.json. "
            "Run the benchmark first with --experiment 15_bandwidth_saturation_sweep."
        )


if __name__ == "__main__":
    main()
