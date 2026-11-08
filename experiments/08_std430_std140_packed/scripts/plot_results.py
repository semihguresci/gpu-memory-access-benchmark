#!/usr/bin/env python3

from __future__ import annotations

import json
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

ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT / "results" / "tables" / "benchmark_results.json"
OUTPUT_PNG = ROOT / "results" / "charts" / "benchmark_summary.png"
EXPERIMENT_ID = "08_std430_std140_packed"


def _parse_notes_field(notes: str) -> dict[str, str]:
    if not notes:
        return {}

    pairs: dict[str, str] = {}
    for token in notes.split(";"):
        chunk = token.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        pairs[key.strip()] = value.strip()
    return pairs


def _extract_note_float(series: pd.Series, key: str) -> pd.Series:
    return pd.to_numeric(series.apply(lambda mapping: mapping.get(key)), errors="coerce")


def _format_problem_size_ticks(values: np.ndarray) -> list[str]:
    labels: list[str] = []
    for value in values:
        count = int(value)
        if count > 0 and count & (count - 1) == 0:
            labels.append(f"2^{count.bit_length() - 1}")
        else:
            labels.append(f"{count:,}")
    return labels


def _plot_std430_std140_packed_summary() -> bool:
    if not INPUT_JSON.exists():
        return False

    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    rows = validated_rows(payload, "08_std430_std140_packed", INPUT_JSON)
    if not rows:
        return False

    df = pd.DataFrame(rows)
    df = df[df.get("experiment_id", "") == EXPERIMENT_ID].copy()
    if df.empty:
        return False

    for column in ["problem_size", "dispatch_count", "gpu_ms", "gbps"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["variant"] = df["variant"].fillna("unknown_variant").astype(str)

    note_maps = df.get("notes", "").fillna("").astype(str).apply(_parse_notes_field)
    df["logical_bytes_per_particle"] = _extract_note_float(note_maps, "logical_bytes_per_particle")
    df["logical_gbps"] = np.where(
        df["gpu_ms"] > 0.0,
        (df["logical_bytes_per_particle"] * df["problem_size"] * df["dispatch_count"] * 2.0) / (df["gpu_ms"] * 1.0e6),
        np.nan,
    )
    df["useful_bandwidth_ratio"] = np.where(df["gbps"] > 0.0, df["logical_gbps"] / df["gbps"], np.nan)

    df = df.dropna(subset=["problem_size", "dispatch_count", "gpu_ms", "gbps", "logical_gbps", "useful_bandwidth_ratio"])
    if df.empty:
        return False

    summary = (
        df.groupby(["variant", "dispatch_count", "problem_size"], as_index=False)
        .agg(
            gpu_ms_median=("gpu_ms", "median"),
            gbps_median=("gbps", "median"),
            logical_gbps_median=("logical_gbps", "median"),
            useful_bandwidth_ratio_median=("useful_bandwidth_ratio", "median"),
        )
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for variant in variants:
        curve = summary[summary["variant"] == variant].sort_values("problem_size")
        axes[0].plot(curve["problem_size"], curve["gpu_ms_median"], marker="o", label=variant)
        axes[1].plot(curve["problem_size"], curve["gbps_median"], marker="o", label=variant)
        axes[2].plot(curve["problem_size"], curve["logical_gbps_median"], marker="o", label=variant)
        axes[3].plot(curve["problem_size"], curve["useful_bandwidth_ratio_median"], marker="o", label=variant)

    axes[0].set_title(f"Dispatch time (median), dispatch_count={representative_dispatch}")
    axes[0].set_ylabel("gpu_ms")
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Effective bandwidth (median, storage bytes)")
    axes[1].set_ylabel("GB/s")
    axes[1].set_xscale("log", base=2)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Useful payload throughput (median, logical bytes)")
    axes[2].set_ylabel("logical GB/s")
    axes[2].set_xscale("log", base=2)
    axes[2].grid(True, alpha=0.3)

    axes[3].set_title("Bandwidth efficiency (logical/storage)")
    axes[3].set_ylabel("ratio")
    axes[3].set_xscale("log", base=2)
    axes[3].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("problem_size (elements)")
        ax.set_xticks(problem_sizes)
        ax.set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)

    axes[1].legend(title="variant")
    fig.suptitle("Experiment 08 std430 vs std140 vs Packed", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    validate_analysis_provenance(INPUT_JSON.parent, "08_std430_std140_packed", required_source=INPUT_JSON)
    if not _plot_std430_std140_packed_summary():
        raise FileNotFoundError(
            "No Experiment 08 rows found in benchmark_results.json. "
            "Run the benchmark first with --experiment 08_std430_std140_packed."
        )
    write_plot_provenance(INPUT_JSON.parent, "08_std430_std140_packed", [OUTPUT_PNG], plot_tools=[Path(__file__)])


if __name__ == "__main__":
    main()
