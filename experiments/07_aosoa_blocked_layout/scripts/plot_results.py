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
EXPERIMENT_ID = "07_aosoa_blocked_layout"


def _format_problem_size_ticks(values: np.ndarray) -> list[str]:
    labels: list[str] = []
    for value in values:
        exponent = int(round(np.log2(value)))
        labels.append(f"2^{exponent}")
    return labels


def _plot_aosoa_summary() -> bool:
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

    for column in ["problem_size", "dispatch_count", "gpu_ms", "gbps"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["variant"] = df["variant"].fillna("unknown_variant").astype(str)
    df = df.dropna(subset=["problem_size", "dispatch_count", "gpu_ms", "gbps"])
    if df.empty:
        return False

    summary = (
        df.groupby(["variant", "dispatch_count", "problem_size"], as_index=False)
        .agg(gpu_ms_median=("gpu_ms", "median"), gbps_median=("gbps", "median"))
        .sort_values(["variant", "dispatch_count", "problem_size"])
    )

    dispatch_counts = sorted(summary["dispatch_count"].unique().tolist())
    if not dispatch_counts:
        return False

    representative_dispatch = dispatch_counts[0]
    summary = summary[summary["dispatch_count"] == representative_dispatch].copy()

    variants = sorted(summary["variant"].unique().tolist())
    problem_sizes = np.array(sorted(summary["problem_size"].unique().tolist()), dtype=float)
    if len(problem_sizes) == 0:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for variant in variants:
        curve = summary[summary["variant"] == variant].sort_values("problem_size")
        axes[0].plot(curve["problem_size"], curve["gpu_ms_median"], marker="o", label=variant)
        axes[1].plot(curve["problem_size"], curve["gbps_median"], marker="o", label=variant)

    axes[0].set_title(f"Dispatch time (median), dispatch_count={representative_dispatch}")
    axes[0].set_ylabel("gpu_ms")
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"Effective bandwidth (median), dispatch_count={representative_dispatch}")
    axes[1].set_ylabel("GB/s")
    axes[1].set_xscale("log", base=2)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("problem_size (elements)")
        ax.set_xticks(problem_sizes)
        ax.set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)

    axes[1].legend(title="variant")
    fig.suptitle("Experiment 07 AoSoA Blocked Layout", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    if not _plot_aosoa_summary():
        raise FileNotFoundError(
            "No Experiment 07 rows found in benchmark_results.json. "
            "Run the benchmark first with --experiment 07_aosoa_blocked_layout."
        )


if __name__ == "__main__":
    main()
