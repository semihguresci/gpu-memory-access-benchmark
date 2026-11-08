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
    validated_rows,
    write_analysis_provenance,
)

EXPERIMENT_ID = "10_scalar_type_width_sweep"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"
INPUT_JSON = TABLES_DIR / "benchmark_results.json"


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


def _ordered_variants(variants: list[str]) -> list[str]:
    order = {
        "fp32": 0,
        "u32": 1,
        "fp16_storage": 2,
        "u16": 3,
        "u8": 4,
    }
    return sorted(variants, key=lambda variant: (order.get(variant, 99), variant))


def _load_rows() -> tuple[pd.DataFrame, dict]:
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            "benchmark_results.json is missing for Experiment 10. "
            "Run with --experiment 10_scalar_type_width_sweep first."
        )

    payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    rows = validated_rows(payload, "10_scalar_type_width_sweep", INPUT_JSON)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input benchmark JSON has no rows[] payload.")

    frame = pd.DataFrame(rows)
    frame = frame[frame.get("experiment_id", "") == EXPERIMENT_ID].copy()
    if frame.empty:
        raise ValueError("Input benchmark JSON has rows but no Experiment 10 entries.")

    for column in ["problem_size", "dispatch_count", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["variant"] = frame["variant"].fillna("unknown_variant").astype(str)
    frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])

    if "notes" not in frame.columns:
        frame["notes"] = ""
    note_maps = frame["notes"].fillna("").astype(str).apply(_parse_notes_field)
    frame["storage_bytes_per_element"] = _extract_note_float(note_maps, "storage_bytes_per_element")
    frame["storage_ratio_vs_fp32"] = _extract_note_float(note_maps, "storage_ratio_vs_fp32")
    frame["validation_tolerance"] = _extract_note_float(note_maps, "validation_tolerance")
    frame["max_abs_error"] = _extract_note_float(note_maps, "max_abs_error")
    frame["mean_abs_error"] = _extract_note_float(note_maps, "mean_abs_error")
    # Missing historical precision measurements stay unavailable, never zero.
    frame["max_abs_error_vs_fp32"] = _extract_note_float(note_maps, "max_abs_error_vs_fp32")
    frame["rms_error_vs_fp32"] = _extract_note_float(note_maps, "rms_error_vs_fp32")
    frame["error_vs_fp32_available"] = frame[["max_abs_error_vs_fp32", "rms_error_vs_fp32"]].notna().all(axis=1)

    metadata = payload.get("metadata", {})
    return frame, metadata


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["variant", "problem_size", "dispatch_count"], as_index=False)
        .agg(
            gpu_ms_median=("gpu_ms", "median"),
            gpu_ms_p95=("gpu_ms", _quantile_95),
            end_to_end_ms_median=("end_to_end_ms", "median"),
            throughput_median=("throughput", "median"),
            gbps_median=("gbps", "median"),
            max_abs_error_median=("max_abs_error", "median"),
            mean_abs_error_median=("mean_abs_error", "median"),
            max_abs_error_vs_fp32_max=("max_abs_error_vs_fp32", "max"),
            rms_error_vs_fp32_median=("rms_error_vs_fp32", "median"),
            error_vs_fp32_available=("error_vs_fp32_available", "all"),
            validation_tolerance_median=("validation_tolerance", "median"),
            storage_bytes_per_element_median=("storage_bytes_per_element", "median"),
            storage_ratio_vs_fp32_median=("storage_ratio_vs_fp32", "median"),
            correctness_pass_rate=("correctness_pass", "mean"),
            sample_count=("gpu_ms", "count"),
        )
        .sort_values(["variant", "problem_size", "dispatch_count"])
        .reset_index(drop=True)
    )
    return summary


def _build_status(rows: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    total_rows = int(rows.shape[0])
    pass_rows = int(rows["correctness_pass"].sum())
    fail_rows = total_rows - pass_rows
    pass_rate = float(pass_rows / total_rows) if total_rows > 0 else 0.0
    largest_problem_size = int(summary["problem_size"].max()) if not summary.empty else 0
    return pd.DataFrame(
        [
            {
                "total_rows": total_rows,
                "correctness_pass_count": pass_rows,
                "correctness_fail_count": fail_rows,
                "correctness_pass_rate": pass_rate,
                "variant_count": int(summary["variant"].nunique()) if not summary.empty else 0,
                "problem_size_count": int(summary["problem_size"].nunique()) if not summary.empty else 0,
                "largest_problem_size": largest_problem_size,
            }
        ]
    )


def _attach_fp32_relative_columns(frame: pd.DataFrame, join_columns: list[str]) -> pd.DataFrame:
    fp32_baseline = (
        frame[frame["variant"] == "fp32"][join_columns + ["gpu_ms_median", "gbps_median"]]
        .rename(
            columns={
                "gpu_ms_median": "fp32_gpu_ms_median",
                "gbps_median": "fp32_gbps_median",
            }
        )
        .copy()
    )

    merged = frame.merge(fp32_baseline, on=join_columns, how="left")
    merged["speedup_vs_fp32"] = merged["fp32_gpu_ms_median"] / merged["gpu_ms_median"]
    merged["delta_gpu_ms_vs_fp32_pct"] = (
        (merged["gpu_ms_median"] - merged["fp32_gpu_ms_median"]) / merged["fp32_gpu_ms_median"]
    ) * 100.0
    merged["delta_gbps_vs_fp32_pct"] = (
        (merged["gbps_median"] - merged["fp32_gbps_median"]) / merged["fp32_gbps_median"]
    ) * 100.0
    return merged


def _build_largest_size_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    largest_problem_size = int(summary["problem_size"].max())
    largest = summary[summary["problem_size"] == largest_problem_size].copy()
    largest["p95_to_median"] = largest["gpu_ms_p95"] / largest["gpu_ms_median"]
    largest = _attach_fp32_relative_columns(largest, ["problem_size", "dispatch_count"])

    largest = largest[
        [
            "variant",
            "problem_size",
            "dispatch_count",
            "gpu_ms_median",
            "gpu_ms_p95",
            "p95_to_median",
            "gbps_median",
            "speedup_vs_fp32",
            "delta_gpu_ms_vs_fp32_pct",
            "delta_gbps_vs_fp32_pct",
            "max_abs_error_median",
            "mean_abs_error_median",
            "max_abs_error_vs_fp32_max",
            "rms_error_vs_fp32_median",
            "error_vs_fp32_available",
        ]
    ].sort_values("gpu_ms_median")

    return largest.reset_index(drop=True)


def _build_speedup_by_size(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    speedup = _attach_fp32_relative_columns(summary.copy(), ["problem_size", "dispatch_count"])
    speedup["p95_to_median"] = speedup["gpu_ms_p95"] / speedup["gpu_ms_median"]

    speedup = speedup[
        [
            "variant",
            "problem_size",
            "dispatch_count",
            "gpu_ms_median",
            "gpu_ms_p95",
            "p95_to_median",
            "gbps_median",
            "speedup_vs_fp32",
            "delta_gpu_ms_vs_fp32_pct",
            "delta_gbps_vs_fp32_pct",
        ]
    ].sort_values(["problem_size", "variant"])

    return speedup.reset_index(drop=True)


def _build_scaling_by_size(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    scaling = summary.copy()
    scaling["p95_to_median"] = scaling["gpu_ms_p95"] / scaling["gpu_ms_median"]
    scaling["gpu_ms_per_million_elements"] = scaling["gpu_ms_median"] / (scaling["problem_size"] / 1.0e6)

    fp32_scaling = (
        scaling[scaling["variant"] == "fp32"][["problem_size", "dispatch_count", "gpu_ms_per_million_elements"]]
        .rename(columns={"gpu_ms_per_million_elements": "fp32_gpu_ms_per_million_elements"})
        .copy()
    )

    scaling = scaling.merge(fp32_scaling, on=["problem_size", "dispatch_count"], how="left")
    scaling["scaling_efficiency_vs_fp32"] = (
        scaling["fp32_gpu_ms_per_million_elements"] / scaling["gpu_ms_per_million_elements"]
    )

    scaling = scaling[
        [
            "variant",
            "problem_size",
            "dispatch_count",
            "gpu_ms_median",
            "gpu_ms_p95",
            "p95_to_median",
            "gpu_ms_per_million_elements",
            "scaling_efficiency_vs_fp32",
            "throughput_median",
            "gbps_median",
        ]
    ].sort_values(["variant", "problem_size"])

    return scaling.reset_index(drop=True)


def _first_problem_size_for_threshold(group: pd.DataFrame, threshold: float) -> int:
    hits = group[group["speedup_vs_fp32"] >= threshold].sort_values("problem_size")
    if hits.empty:
        return -1
    return int(hits.iloc[0]["problem_size"])


def _build_speedup_crossover(speedup_by_size: pd.DataFrame) -> pd.DataFrame:
    if speedup_by_size.empty:
        return pd.DataFrame()

    records: list[dict] = []
    for variant, group in speedup_by_size.groupby("variant"):
        ordered = group.sort_values("problem_size").reset_index(drop=True)
        max_index = int(ordered["speedup_vs_fp32"].idxmax())
        max_row = ordered.loc[max_index]
        records.append(
            {
                "variant": str(variant),
                "first_size_speedup_ge_1_0": _first_problem_size_for_threshold(ordered, 1.0),
                "first_size_speedup_ge_1_5": _first_problem_size_for_threshold(ordered, 1.5),
                "first_size_speedup_ge_2_0": _first_problem_size_for_threshold(ordered, 2.0),
                "max_speedup_vs_fp32": float(max_row["speedup_vs_fp32"]),
                "size_at_max_speedup": int(max_row["problem_size"]),
            }
        )

    crossover = pd.DataFrame(records)
    crossover = crossover.sort_values(
        by=["max_speedup_vs_fp32", "variant"], ascending=[False, True]
    ).reset_index(drop=True)
    return crossover


def _build_stability_overview(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    working = summary.copy()
    working["p95_to_median"] = working["gpu_ms_p95"] / working["gpu_ms_median"]
    largest_problem_size = int(working["problem_size"].max())

    records: list[dict] = []
    for variant, group in working.groupby("variant"):
        group = group.sort_values("problem_size").reset_index(drop=True)
        max_index = int(group["p95_to_median"].idxmax())
        max_row = group.loc[max_index]
        largest_group = group[group["problem_size"] == largest_problem_size]
        largest_ratio = float(largest_group["p95_to_median"].iloc[0]) if not largest_group.empty else float("nan")

        records.append(
            {
                "variant": str(variant),
                "avg_p95_to_median": float(group["p95_to_median"].mean()),
                "max_p95_to_median": float(max_row["p95_to_median"]),
                "worst_problem_size": int(max_row["problem_size"]),
                "largest_size_p95_to_median": largest_ratio,
            }
        )

    stability = pd.DataFrame(records)
    stability = stability.sort_values("avg_p95_to_median").reset_index(drop=True)
    return stability


def _plot_line_metric(summary: pd.DataFrame, metric_column: str, y_label: str, title: str, output_path: Path) -> Path | None:
    plot_frame = summary.copy()
    dispatch_counts = sorted(plot_frame["dispatch_count"].unique().tolist())
    if not dispatch_counts:
        return
    representative_dispatch = dispatch_counts[0]
    plot_frame = plot_frame[plot_frame["dispatch_count"] == representative_dispatch]
    if plot_frame.empty:
        return

    variants = _ordered_variants(plot_frame["variant"].unique().tolist())
    problem_sizes = np.array(sorted(plot_frame["problem_size"].unique().tolist()), dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    for variant in variants:
        curve = plot_frame[plot_frame["variant"] == variant].sort_values("problem_size")
        ax.plot(curve["problem_size"], curve[metric_column], marker="o", linewidth=2.0, label=variant)

    ax.set_xscale("log", base=2)
    ax.set_xticks(problem_sizes)
    ax.set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)
    ax.set_xlabel("problem_size (elements)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} (dispatch_count={representative_dispatch})")
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_speedup_by_size(speedup_by_size: pd.DataFrame, output_path: Path) -> Path | None:
    if speedup_by_size.empty:
        return

    plot_frame = speedup_by_size.copy()
    dispatch_counts = sorted(plot_frame["dispatch_count"].unique().tolist())
    if not dispatch_counts:
        return
    representative_dispatch = dispatch_counts[0]
    plot_frame = plot_frame[plot_frame["dispatch_count"] == representative_dispatch]
    if plot_frame.empty:
        return

    variants = _ordered_variants(plot_frame["variant"].unique().tolist())
    problem_sizes = np.array(sorted(plot_frame["problem_size"].unique().tolist()), dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    for variant in variants:
        curve = plot_frame[plot_frame["variant"] == variant].sort_values("problem_size")
        ax.plot(curve["problem_size"], curve["speedup_vs_fp32"], marker="o", linewidth=2.0, label=variant)

    ax.set_xscale("log", base=2)
    ax.set_xticks(problem_sizes)
    ax.set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)
    ax.set_xlabel("problem_size (elements)")
    ax.set_ylabel("speedup vs fp32 (higher is faster)")
    ax.set_title(f"Experiment 10: Relative Runtime vs fp32 (dispatch_count={representative_dispatch})")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_stability_ratio(speedup_by_size: pd.DataFrame, output_path: Path) -> Path | None:
    if speedup_by_size.empty:
        return

    plot_frame = speedup_by_size.copy()
    dispatch_counts = sorted(plot_frame["dispatch_count"].unique().tolist())
    if not dispatch_counts:
        return
    representative_dispatch = dispatch_counts[0]
    plot_frame = plot_frame[plot_frame["dispatch_count"] == representative_dispatch]
    if plot_frame.empty:
        return

    variants = _ordered_variants(plot_frame["variant"].unique().tolist())
    problem_sizes = np.array(sorted(plot_frame["problem_size"].unique().tolist()), dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    for variant in variants:
        curve = plot_frame[plot_frame["variant"] == variant].sort_values("problem_size")
        ax.plot(curve["problem_size"], curve["p95_to_median"], marker="o", linewidth=2.0, label=variant)

    ax.set_xscale("log", base=2)
    ax.set_xticks(problem_sizes)
    ax.set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)
    ax.set_xlabel("problem_size (elements)")
    ax.set_ylabel("p95 / median (lower is steadier)")
    ax.set_title(f"Experiment 10: Runtime Stability (dispatch_count={representative_dispatch})")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_scaling_efficiency(scaling_by_size: pd.DataFrame, output_path: Path) -> Path | None:
    if scaling_by_size.empty:
        return

    plot_frame = scaling_by_size.copy()
    dispatch_counts = sorted(plot_frame["dispatch_count"].unique().tolist())
    if not dispatch_counts:
        return
    representative_dispatch = dispatch_counts[0]
    plot_frame = plot_frame[plot_frame["dispatch_count"] == representative_dispatch]
    if plot_frame.empty:
        return

    variants = _ordered_variants(plot_frame["variant"].unique().tolist())
    problem_sizes = np.array(sorted(plot_frame["problem_size"].unique().tolist()), dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    for variant in variants:
        curve = plot_frame[plot_frame["variant"] == variant].sort_values("problem_size")
        ax.plot(curve["problem_size"], curve["gpu_ms_per_million_elements"], marker="o", linewidth=2.0, label=variant)

    ax.set_xscale("log", base=2)
    ax.set_xticks(problem_sizes)
    ax.set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)
    ax.set_xlabel("problem_size (elements)")
    ax.set_ylabel("gpu_ms per million elements (lower is better)")
    ax.set_title(f"Experiment 10: Scaling Efficiency (dispatch_count={representative_dispatch})")
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_error_bars(summary: pd.DataFrame, output_path: Path) -> Path | None:
    if summary.empty:
        return

    largest = summary.sort_values("problem_size").groupby("variant", as_index=False).tail(1)
    largest = largest.sort_values("variant")
    if largest.empty:
        return

    variants = largest["variant"].tolist()
    x = np.arange(len(variants))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - (width / 2.0), largest["max_abs_error_median"], width=width, label="max format residual")
    ax.bar(x + (width / 2.0), largest["mean_abs_error_median"], width=width, label="mean format residual")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("absolute residual versus representation-specific oracle")
    ax.set_title("Experiment 10: Implementation Correctness Residual at Largest Problem Size")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_fp32_error_bars(summary: pd.DataFrame, output_path: Path) -> Path | None:
    largest = summary[summary["problem_size"] == summary["problem_size"].max()].copy()
    largest = largest.sort_values("variant")
    fig, ax = plt.subplots(figsize=(11, 6))
    if largest.empty or not largest["error_vs_fp32_available"].all():
        ax.text(0.5, 0.5, "Error versus fp32 is unavailable for this run.", ha="center", va="center")
        ax.set_axis_off()
        print("[info] Common fp32 error unavailable in this run; format residual is a separate metric.")
    else:
        x = np.arange(len(largest))
        width = 0.36
        ax.bar(x - width / 2.0, largest["max_abs_error_vs_fp32_max"], width=width, label="max absolute error vs fp32")
        ax.bar(x + width / 2.0, largest["rms_error_vs_fp32_median"], width=width, label="RMS error vs fp32 (median)")
        ax.set_xticks(x)
        ax.set_xticklabels(largest["variant"])
        ax.set_ylabel("absolute error versus common fp32 transform")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
    ax.set_title("Experiment 10: Numerical Precision at Largest Problem Size")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    generated_tables: list[Path] = []
    generated_charts: list[Path | None] = []
    rows, metadata = _load_rows()
    summary = _build_summary(rows)
    status = _build_status(rows, summary)
    largest_size_comparison = _build_largest_size_comparison(summary)
    speedup_by_size = _build_speedup_by_size(summary)
    scaling_by_size = _build_scaling_by_size(summary)
    speedup_crossover = _build_speedup_crossover(speedup_by_size)
    stability_overview = _build_stability_overview(summary)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = TABLES_DIR / "scalar_type_width_sweep_summary.csv"
    status_path = TABLES_DIR / "scalar_type_width_sweep_status_overview.csv"
    largest_size_path = TABLES_DIR / "scalar_type_width_sweep_largest_size_comparison.csv"
    speedup_path = TABLES_DIR / "scalar_type_width_sweep_speedup_vs_fp32_by_size.csv"
    scaling_path = TABLES_DIR / "scalar_type_width_sweep_scaling_by_size.csv"
    crossover_path = TABLES_DIR / "scalar_type_width_sweep_speedup_crossover.csv"
    stability_path = TABLES_DIR / "scalar_type_width_sweep_stability_overview.csv"

    summary.to_csv(summary_path, index=False)
    generated_tables.append(summary_path)
    status.to_csv(status_path, index=False)
    generated_tables.append(status_path)
    largest_size_comparison.to_csv(largest_size_path, index=False)
    generated_tables.append(largest_size_path)
    speedup_by_size.to_csv(speedup_path, index=False)
    generated_tables.append(speedup_path)
    scaling_by_size.to_csv(scaling_path, index=False)
    generated_tables.append(scaling_path)
    speedup_crossover.to_csv(crossover_path, index=False)
    generated_tables.append(crossover_path)
    stability_overview.to_csv(stability_path, index=False)
    generated_tables.append(stability_path)

    generated_charts.append(_plot_line_metric(
        summary,
        metric_column="gpu_ms_median",
        y_label="gpu_ms (median)",
        title="Experiment 10: Dispatch Time",
        output_path=CHARTS_DIR / "scalar_type_width_sweep_median_gpu_ms.png",
    ))
    generated_charts.append(_plot_line_metric(
        summary,
        metric_column="gbps_median",
        y_label="GB/s (median)",
        title="Experiment 10: Effective Bandwidth",
        output_path=CHARTS_DIR / "scalar_type_width_sweep_median_gbps.png",
    ))
    generated_charts.append(_plot_speedup_by_size(
        speedup_by_size,
        output_path=CHARTS_DIR / "scalar_type_width_sweep_speedup_vs_fp32.png",
    ))
    generated_charts.append(_plot_stability_ratio(
        speedup_by_size,
        output_path=CHARTS_DIR / "scalar_type_width_sweep_p95_to_median_ratio.png",
    ))
    generated_charts.append(_plot_scaling_efficiency(
        scaling_by_size,
        output_path=CHARTS_DIR / "scalar_type_width_sweep_gpu_ms_per_million_elements.png",
    ))
    generated_charts.append(_plot_error_bars(
        summary,
        output_path=CHARTS_DIR / "scalar_type_width_sweep_error_metrics.png",
    ))
    generated_charts.append(_plot_fp32_error_bars(summary, CHARTS_DIR / "scalar_type_width_sweep_error_vs_fp32.png"))

    write_analysis_provenance(TABLES_DIR, [INPUT_JSON], EXPERIMENT_ID, derived_tables=generated_tables, derived_charts=generated_charts, analysis_tools=[Path(__file__)])
    print(f"rows={len(rows)}")
    print(f"summary_rows={len(summary)}")
    print(f"gpu={metadata.get('gpu_name', 'unknown')}")
    print(f"wrote={summary_path}")
    print(f"wrote={status_path}")
    print(f"wrote={largest_size_path}")
    print(f"wrote={speedup_path}")
    print(f"wrote={scaling_path}")
    print(f"wrote={crossover_path}")
    print(f"wrote={stability_path}")
    print(f"wrote={CHARTS_DIR / 'scalar_type_width_sweep_median_gpu_ms.png'}")
    print(f"wrote={CHARTS_DIR / 'scalar_type_width_sweep_median_gbps.png'}")
    print(f"wrote={CHARTS_DIR / 'scalar_type_width_sweep_speedup_vs_fp32.png'}")
    print(f"wrote={CHARTS_DIR / 'scalar_type_width_sweep_p95_to_median_ratio.png'}")
    print(f"wrote={CHARTS_DIR / 'scalar_type_width_sweep_gpu_ms_per_million_elements.png'}")
    print(f"wrote={CHARTS_DIR / 'scalar_type_width_sweep_error_metrics.png'}")
    print(f"wrote={CHARTS_DIR / 'scalar_type_width_sweep_error_vs_fp32.png'}")


if __name__ == "__main__":
    main()
