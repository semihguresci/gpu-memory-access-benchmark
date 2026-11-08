import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analysis_integrity import (
    validate_analysis_provenance,
    validated_rows,
    write_plot_provenance,
)
import json

ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT / "results" / "tables" / "benchmark_results.json"
INPUT_CSV = ROOT / "results" / "tables" / "benchmark_results.csv"
OUTPUT_PNG = ROOT / "results" / "charts" / "benchmark_summary.png"


def _format_problem_size_ticks(values: np.ndarray) -> list[str]:
    labels: list[str] = []
    for value in values:
        count = int(value)
        if count > 0 and count & (count - 1) == 0:
            labels.append(f"2^{count.bit_length() - 1}")
        else:
            labels.append(f"{count:,}")
    return labels


def _plot_dispatch_basics_summary() -> bool:
    if not INPUT_JSON.exists():
        return False

    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    rows = validated_rows(payload, "01_dispatch_basics", INPUT_JSON)
    if not rows:
        return False

    df = pd.DataFrame(rows)
    df = df[df["experiment_id"] == "01_dispatch_basics"].copy()
    if df.empty:
        return False

    numeric_columns = ["problem_size", "dispatch_count", "gpu_ms", "throughput"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=numeric_columns)

    summary = (
        df.groupby(["variant", "problem_size", "dispatch_count"], as_index=False)
        .agg(gpu_ms_median=("gpu_ms", "median"), throughput_median=("throughput", "median"))
        .sort_values(["variant", "problem_size", "dispatch_count"])
    )

    variants = sorted(summary["variant"].unique().tolist())
    dispatch_counts = sorted(summary["dispatch_count"].unique().tolist())
    problem_sizes = np.array(sorted(summary["problem_size"].unique().tolist()), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    for row_idx, variant in enumerate(variants):
        subset = summary[summary["variant"] == variant]

        for dispatch_count in dispatch_counts:
            curve = subset[subset["dispatch_count"] == dispatch_count].sort_values("problem_size")
            if curve.empty:
                continue

            axes[row_idx, 0].plot(curve["problem_size"], curve["gpu_ms_median"], marker="o", label=f"d={dispatch_count}")
            axes[row_idx, 1].plot(
                curve["problem_size"], curve["throughput_median"] / 1.0e9, marker="o", label=f"d={dispatch_count}"
            )

        axes[row_idx, 0].set_title(f"{variant}: Dispatch GPU Time (median)")
        axes[row_idx, 0].set_ylabel("gpu_ms")
        axes[row_idx, 0].grid(True, alpha=0.3)
        axes[row_idx, 0].set_xscale("log", base=2)

        axes[row_idx, 1].set_title(f"{variant}: Throughput (median)")
        axes[row_idx, 1].set_ylabel("throughput (Gelem/s)")
        axes[row_idx, 1].grid(True, alpha=0.3)
        axes[row_idx, 1].set_xscale("log", base=2)

    for col_idx in range(2):
        axes[1, col_idx].set_xlabel("problem_size (elements)")
        axes[1, col_idx].set_xticks(problem_sizes)
        axes[1, col_idx].set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)

    axes[0, 0].legend(title="dispatch_count", ncol=2)
    axes[0, 1].legend(title="dispatch_count", ncol=2)
    fig.suptitle("Experiment 01 Dispatch Basics: Problem Size x Dispatch Count x Operation", fontsize=14)
    fig.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    return True


def _plot_generic_csv_summary() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["experiment"], df["average_ms"])
    ax.set_title("Benchmark Average Time")
    ax.set_ylabel("Average Time (ms)")
    ax.set_xlabel("Experiment")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)


def main() -> None:
    validate_analysis_provenance(INPUT_JSON.parent, "01_dispatch_basics", required_source=INPUT_JSON)
    if _plot_dispatch_basics_summary():
        write_plot_provenance(INPUT_JSON.parent, "01_dispatch_basics", [OUTPUT_PNG], plot_tools=[Path(__file__)])
        return
    raise ValueError("No validated Experiment 01 raw rows are available for plotting.")


if __name__ == "__main__":
    main()
