#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT / "results" / "tables" / "benchmark_results.json"
OUTPUT_PNG = ROOT / "results" / "charts" / "benchmark_summary.png"
EXPERIMENT_ID = "02_local_size_sweep"
LOCAL_SIZE_RE = re.compile(r"_ls(\d+)$")


def _extract_local_size(variant: str) -> int | None:
    match = LOCAL_SIZE_RE.search(variant)
    if match is None:
        return None
    return int(match.group(1))


def _base_variant(variant: str) -> str:
    return LOCAL_SIZE_RE.sub("", variant)


def _plot_local_size_sweep_summary() -> bool:
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

    numeric_columns = ["problem_size", "dispatch_count", "gpu_ms", "throughput"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["variant"] = df["variant"].fillna("unknown_variant").astype(str)
    df["local_size_x"] = df["variant"].apply(_extract_local_size)
    df["base_variant"] = df["variant"].apply(_base_variant)
    df = df.dropna(subset=["problem_size", "dispatch_count", "gpu_ms", "throughput", "local_size_x"])
    if df.empty:
        return False

    df["problem_size"] = df["problem_size"].astype(int)
    df["dispatch_count"] = df["dispatch_count"].astype(int)
    df["local_size_x"] = df["local_size_x"].astype(int)

    summary = (
        df.groupby(["base_variant", "problem_size", "dispatch_count", "local_size_x"], as_index=False)
        .agg(gpu_ms_median=("gpu_ms", "median"), throughput_median=("throughput", "median"))
        .sort_values(["base_variant", "problem_size", "dispatch_count", "local_size_x"])
    )
    if summary.empty:
        return False

    variants = sorted(summary["base_variant"].unique().tolist())
    fig, axes = plt.subplots(len(variants), 2, figsize=(16, 4.5 * len(variants)), sharex=True)
    if len(variants) == 1:
        axes = np.array([axes])

    for row_idx, variant in enumerate(variants):
        subset = summary[summary["base_variant"] == variant].copy()

        for (dispatch_count, problem_size), curve in subset.groupby(["dispatch_count", "problem_size"], sort=True):
            curve = curve.sort_values("local_size_x")
            label = f"d={int(dispatch_count)}, size=2^{int(round(np.log2(problem_size)))}"
            axes[row_idx, 0].plot(curve["local_size_x"], curve["gpu_ms_median"], marker="o", label=label)
            axes[row_idx, 1].plot(
                curve["local_size_x"],
                curve["throughput_median"] / 1.0e9,
                marker="o",
                label=label,
            )

        axes[row_idx, 0].set_title(f"{variant}: GPU time vs local_size_x")
        axes[row_idx, 0].set_ylabel("gpu_ms (median)")
        axes[row_idx, 0].set_xscale("log", base=2)
        axes[row_idx, 0].grid(True, alpha=0.3)

        axes[row_idx, 1].set_title(f"{variant}: Throughput vs local_size_x")
        axes[row_idx, 1].set_ylabel("throughput (Gelem/s)")
        axes[row_idx, 1].set_xscale("log", base=2)
        axes[row_idx, 1].grid(True, alpha=0.3)

    local_sizes = np.array(sorted(summary["local_size_x"].unique().tolist()), dtype=float)
    for col_idx in range(2):
        axes[-1, col_idx].set_xlabel("local_size_x")
        axes[-1, col_idx].set_xticks(local_sizes)
        axes[-1, col_idx].set_xticklabels([str(int(v)) for v in local_sizes])
        axes[0, col_idx].legend(title="dispatch/problem", ncol=2, fontsize=8)

    fig.suptitle("Experiment 02 Local Size Sweep Summary", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    if not _plot_local_size_sweep_summary():
        raise FileNotFoundError(
            "No Experiment 02 rows found in benchmark_results.json. "
            "Run the benchmark first with --experiment 02_local_size_sweep."
        )


if __name__ == "__main__":
    main()
