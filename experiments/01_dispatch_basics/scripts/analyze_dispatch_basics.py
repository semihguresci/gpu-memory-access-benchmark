#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analysis_integrity import (
    measurement_revision,
    preferred_run_id,
    select_compatible_runs,
    validated_rows,
    write_analysis_provenance,
)

EXPERIMENT_ID = "01_dispatch_basics"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"
CURRENT_RUN_JSON = TABLES_DIR / "benchmark_results.json"
DEVICE_COLUMNS = ["gpu_name", "driver_version", "vulkan_api_version", "device_id", "device_slug"]
RUN_CONFIGURATION_COLUMNS = ["measurement_revision", "validation_enabled", "warmup_iterations", "timed_iterations"]


def _slugify(value: str, fallback: str = "unknown") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or fallback


def _parse_iso8601(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_problem_size_ticks(values: np.ndarray) -> list[str]:
    labels: list[str] = []
    for value in values:
        count = int(value)
        if count > 0 and count & (count - 1) == 0:
            labels.append(f"2^{count.bit_length() - 1}")
        else:
            labels.append(f"{count:,}")
    return labels


def _discover_run_files(runs_dir: Path, include_current: bool, revision: str | None = None) -> list[Path]:
    paths = sorted(runs_dir.rglob("*.json")) if runs_dir.exists() else []
    if include_current and CURRENT_RUN_JSON.exists():
        current_resolved = CURRENT_RUN_JSON.resolve()
        if all(path.resolve() != current_resolved for path in paths):
            paths.append(CURRENT_RUN_JSON)
    return select_compatible_runs(paths, current_path=CURRENT_RUN_JSON if include_current else None, revision=revision)


def _relative_run_id(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _build_run_record(path: Path, payload: dict, frame: pd.DataFrame) -> dict:
    metadata = payload.get("metadata", {})
    exported_at_utc = str(metadata.get("exported_at_utc", ""))
    exported_at_dt = _parse_iso8601(exported_at_utc)
    if exported_at_dt is None:
        exported_at_dt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    driver_version = str(metadata.get("driver_version", "unknown_driver"))
    vulkan_api_version = str(metadata.get("vulkan_api_version", "unknown_api"))
    validation_enabled = bool(metadata.get("validation_enabled", False))
    warmup_iterations = int(metadata.get("warmup_iterations", 0))
    timed_iterations = int(metadata.get("timed_iterations", 0))

    device_id = f"{gpu_name} | drv {driver_version} | vk {vulkan_api_version}"
    device_slug = _slugify(f"{gpu_name}_{driver_version}_{vulkan_api_version}")
    run_id = _relative_run_id(path)

    return {
        "measurement_revision": measurement_revision(metadata),
        "run_id": run_id,
        "run_file": run_id,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "vulkan_api_version": vulkan_api_version,
        "validation_enabled": validation_enabled,
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "exported_at_utc": exported_at_utc or exported_at_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exported_at_dt": exported_at_dt,
        "device_id": device_id,
        "device_slug": device_slug,
        "row_count": int(frame.shape[0]),
        "correctness_pass_rate": float(frame["correctness_pass"].mean()),
        "problem_size_min": int(frame["problem_size"].min()),
        "problem_size_max": int(frame["problem_size"].max()),
        "dispatch_counts": ",".join(str(int(value)) for value in sorted(frame["dispatch_count"].unique())),
    }


def _load_dispatch_rows(path: Path) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] Skipping unreadable JSON: {path} ({exc})")
        return None, None

    rows = validated_rows(payload, "01_dispatch_basics", path)
    if not isinstance(rows, list) or not rows:
        return None, None

    frame = pd.DataFrame(rows)
    if "experiment_id" not in frame.columns:
        return None, None

    frame = frame[frame["experiment_id"] == EXPERIMENT_ID].copy()
    if frame.empty:
        return None, None

    numeric_columns = ["problem_size", "dispatch_count", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    frame = frame.dropna(subset=["problem_size", "dispatch_count", "gpu_ms", "throughput", "gbps"])
    if frame.empty:
        return None, None

    frame["problem_size"] = frame["problem_size"].astype(int)
    frame["dispatch_count"] = frame["dispatch_count"].astype(int)
    frame["iteration"] = frame["iteration"].fillna(0).astype(int)
    frame["variant"] = frame["variant"].fillna("unknown_variant").astype(str)
    if "notes" in frame.columns:
        frame["notes"] = frame["notes"].fillna("").astype(str)
    else:
        frame["notes"] = ""

    if "correctness_pass" in frame.columns:
        frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
    else:
        frame["correctness_pass"] = False

    run_record = _build_run_record(path, payload, frame)
    for key in [
        "measurement_revision",
        "run_id",
        "run_file",
        "gpu_name",
        "driver_version",
        "vulkan_api_version",
        "validation_enabled",
        "warmup_iterations",
        "timed_iterations",
        "exported_at_utc",
        "exported_at_dt",
        "device_id",
        "device_slug",
    ]:
        frame[key] = run_record[key]

    return frame, run_record


def _load_all_runs(paths: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    run_records: list[dict] = []

    # Discovery deduplicates complete payloads; timestamps do not identify runs.
    for path in paths:
        frame, run_record = _load_dispatch_rows(path)
        if frame is None or run_record is None:
            continue
        frames.append(frame)
        run_records.append(run_record)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    all_rows = pd.concat(frames, ignore_index=True)
    run_index = pd.DataFrame(run_records)
    run_index = run_index.sort_values(["exported_at_dt", "run_id"]).reset_index(drop=True)
    return all_rows, run_index


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_run_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "measurement_revision",
        "run_id",
        "run_file",
        "gpu_name",
        "driver_version",
        "vulkan_api_version",
        "validation_enabled",
        "warmup_iterations",
        "timed_iterations",
        "exported_at_utc",
        "device_id",
        "device_slug",
        "variant",
        "problem_size",
        "dispatch_count",
    ]
    summary = (
        all_rows.groupby(group_columns, as_index=False)
        .agg(
            gpu_ms_median=("gpu_ms", "median"),
            gpu_ms_p95=("gpu_ms", _quantile_95),
            end_to_end_ms_median=("end_to_end_ms", "median"),
            throughput_median=("throughput", "median"),
            throughput_p95=("throughput", _quantile_95),
            gbps_median=("gbps", "median"),
            correctness_pass_rate=("correctness_pass", "mean"),
            sample_count=("gpu_ms", "count"),
        )
        .sort_values(["exported_at_utc", "run_id", "variant", "problem_size", "dispatch_count"])
        .reset_index(drop=True)
    )
    return summary


def _best_dispatch_table(summary: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records: list[dict] = []
    for group_key, group in summary.groupby(group_columns, sort=True):
        clean_group = group.dropna(subset=["gpu_ms_median", "throughput_median"])
        if clean_group.empty:
            continue

        latency_row = clean_group.loc[clean_group["gpu_ms_median"].idxmin()]
        throughput_row = clean_group.loc[clean_group["throughput_median"].idxmax()]

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        record = {column: value for column, value in zip(group_columns, group_key)}
        record.update(
            {
                "best_dispatch_for_latency": int(latency_row["dispatch_count"]),
                "best_gpu_ms": float(latency_row["gpu_ms_median"]),
                "best_dispatch_for_throughput": int(throughput_row["dispatch_count"]),
                "best_throughput": float(throughput_row["throughput_median"]),
            }
        )
        records.append(record)

    return pd.DataFrame(records)


def _write_table(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def _plot_time_throughput(summary: pd.DataFrame, output_path: Path, title: str) -> Path | None:
    if summary.empty:
        return

    variants = sorted(summary["variant"].unique().tolist())
    dispatch_counts = sorted(summary["dispatch_count"].unique().tolist())
    problem_sizes = np.array(sorted(summary["problem_size"].unique().tolist()), dtype=float)
    if len(problem_sizes) == 0:
        return

    fig, axes = plt.subplots(len(variants), 2, figsize=(16, 4.5 * len(variants)), sharex=True)
    if len(variants) == 1:
        axes = np.array([axes])

    for row_index, variant in enumerate(variants):
        subset = summary[summary["variant"] == variant]

        for dispatch_count in dispatch_counts:
            curve = subset[subset["dispatch_count"] == dispatch_count].sort_values("problem_size")
            if curve.empty:
                continue

            axes[row_index, 0].plot(
                curve["problem_size"],
                curve["gpu_ms_median"],
                marker="o",
                label=f"d={dispatch_count}",
            )
            axes[row_index, 1].plot(
                curve["problem_size"],
                curve["throughput_median"] / 1.0e9,
                marker="o",
                label=f"d={dispatch_count}",
            )

        axes[row_index, 0].set_title(f"{variant}: Dispatch GPU Time (median)")
        axes[row_index, 0].set_ylabel("gpu_ms")
        axes[row_index, 0].set_xscale("log", base=2)
        axes[row_index, 0].grid(True, alpha=0.3)

        axes[row_index, 1].set_title(f"{variant}: Throughput (median)")
        axes[row_index, 1].set_ylabel("throughput (Gelem/s)")
        axes[row_index, 1].set_xscale("log", base=2)
        axes[row_index, 1].grid(True, alpha=0.3)

    for column_index in range(2):
        axes[-1, column_index].set_xlabel("problem_size (elements)")
        axes[-1, column_index].set_xticks(problem_sizes)
        axes[-1, column_index].set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)
        axes[0, column_index].legend(title="dispatch_count", ncol=2)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_cross_device_comparison(device_summary: pd.DataFrame, output_path: Path) -> Path | None:
    if device_summary.empty:
        return

    variant = "contiguous_write"
    if variant not in set(device_summary["variant"].unique().tolist()):
        variant = sorted(device_summary["variant"].unique().tolist())[0]

    subset = device_summary[device_summary["variant"] == variant].copy()
    if subset.empty:
        return

    dispatch_counts = sorted(subset["dispatch_count"].unique().tolist())
    low_dispatch = dispatch_counts[0]
    high_dispatch = dispatch_counts[-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    for device_label, _, device_frame in _device_configuration_groups(subset):
        low_curve = device_frame[device_frame["dispatch_count"] == low_dispatch].sort_values("problem_size")
        high_curve = device_frame[device_frame["dispatch_count"] == high_dispatch].sort_values("problem_size")
        if low_curve.empty or high_curve.empty:
            continue

        axes[0].plot(low_curve["problem_size"], low_curve["gpu_ms_median"], marker="o", label=device_label)
        axes[1].plot(
            high_curve["problem_size"],
            high_curve["throughput_median"] / 1.0e9,
            marker="o",
            label=device_label,
        )

    problem_sizes = np.array(sorted(subset["problem_size"].unique().tolist()), dtype=float)
    if len(problem_sizes) == 0:
        plt.close(fig)
        return

    axes[0].set_title(f"{variant}: latency profile (dispatch_count={low_dispatch})")
    axes[0].set_ylabel("gpu_ms")
    axes[0].set_xlabel("problem_size (elements)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(problem_sizes)
    axes[0].set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="device and run settings", fontsize=8)

    axes[1].set_title(f"{variant}: throughput profile (dispatch_count={high_dispatch})")
    axes[1].set_ylabel("throughput (Gelem/s)")
    axes[1].set_xlabel("problem_size (elements)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(problem_sizes)
    axes[1].set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title="device and run settings", fontsize=8)

    fig.suptitle("Experiment 01 Cross-Device Comparison", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_device_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    group_columns = DEVICE_COLUMNS + RUN_CONFIGURATION_COLUMNS + ["variant", "problem_size", "dispatch_count"]
    device_summary = (
        run_summary.groupby(group_columns, as_index=False)
        .agg(
            gpu_ms_median=("gpu_ms_median", "median"),
            gpu_ms_p95=("gpu_ms_median", _quantile_95),
            throughput_median=("throughput_median", "median"),
            throughput_p95=("throughput_median", _quantile_95),
            gbps_median=("gbps_median", "median"),
            correctness_pass_rate=("correctness_pass_rate", "mean"),
            run_count=("run_id", "nunique"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return device_summary


def _build_operation_ratio_table(device_summary: pd.DataFrame) -> pd.DataFrame:
    variants = set(device_summary["variant"].unique().tolist())
    if "contiguous_write" not in variants or "noop" not in variants:
        return pd.DataFrame()

    write_frame = device_summary[device_summary["variant"] == "contiguous_write"].copy()
    noop_frame = device_summary[device_summary["variant"] == "noop"].copy()
    join_columns = DEVICE_COLUMNS + RUN_CONFIGURATION_COLUMNS + ["problem_size", "dispatch_count"]
    merged = write_frame.merge(
        noop_frame,
        on=join_columns,
        suffixes=("_write", "_noop"),
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["gpu_ms_ratio_write_over_noop"] = merged["gpu_ms_median_write"] / merged["gpu_ms_median_noop"]
    merged["throughput_ratio_write_over_noop"] = (
        merged["throughput_median_write"] / merged["throughput_median_noop"].where(merged["throughput_median_noop"] > 0.0)
    )

    columns = join_columns + [
        "gpu_ms_ratio_write_over_noop",
        "throughput_ratio_write_over_noop",
    ]
    return merged[columns].sort_values(join_columns).reset_index(drop=True)


def _plot_operation_ratio_by_device(operation_ratios: pd.DataFrame, output_path: Path) -> Path | None:
    if operation_ratios.empty:
        return

    aggregate = (
        operation_ratios.groupby(DEVICE_COLUMNS + RUN_CONFIGURATION_COLUMNS + ["dispatch_count"], as_index=False)
        .agg(
            gpu_ms_ratio_write_over_noop=("gpu_ms_ratio_write_over_noop", "median"),
            throughput_ratio_write_over_noop=("throughput_ratio_write_over_noop", "median"),
        )
        .sort_values(DEVICE_COLUMNS + RUN_CONFIGURATION_COLUMNS + ["dispatch_count"])
    )
    if aggregate.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    for device_label, _, frame in _device_configuration_groups(aggregate):
        axes[0].plot(
            frame["dispatch_count"],
            frame["gpu_ms_ratio_write_over_noop"],
            marker="o",
            label=device_label,
        )
        if frame["throughput_ratio_write_over_noop"].notna().any():
            axes[1].plot(
                frame["dispatch_count"],
                frame["throughput_ratio_write_over_noop"],
                marker="o",
                label=device_label,
            )

    axes[0].set_title("GPU time ratio (contiguous_write / noop)")
    axes[0].set_xlabel("dispatch_count")
    axes[0].set_ylabel("ratio")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="device and run settings", fontsize=8)

    if not aggregate["throughput_ratio_write_over_noop"].notna().any():
        axes[1].text(0.5, 0.5, "Undefined: no-op has no useful payload.", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title("Throughput ratio (contiguous_write / noop)")
    axes[1].set_xlabel("dispatch_count")
    axes[1].set_ylabel("ratio")
    axes[1].grid(True, alpha=0.3)
    if aggregate["throughput_ratio_write_over_noop"].notna().any():
        axes[1].legend(title="device and run settings", fontsize=8)

    fig.suptitle("Experiment 01 Operation Cost Delta by Device", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_representative_points_table(latest_summary: pd.DataFrame) -> pd.DataFrame:
    if latest_summary.empty:
        return pd.DataFrame()

    target_variant = "contiguous_write"
    target_problem_sizes = [2**10, 2**18, 2**20]
    target_dispatch_counts = [1, 64]

    selection = latest_summary[
        (latest_summary["variant"] == target_variant)
        & (latest_summary["problem_size"].isin(target_problem_sizes))
        & (latest_summary["dispatch_count"].isin(target_dispatch_counts))
    ].copy()
    if selection.empty:
        return pd.DataFrame()

    selection = selection.sort_values(["problem_size", "dispatch_count"]).reset_index(drop=True)
    selection["problem_size_label"] = selection["problem_size"].apply(
        lambda value: f"2^{int(round(np.log2(value)))} ({int(value)})"
    )
    selection["point_label"] = selection.apply(
        lambda row: f"{row['problem_size_label']}\nd={int(row['dispatch_count'])}",
        axis=1,
    )
    return selection[
        [
            "variant",
            "problem_size",
            "problem_size_label",
            "dispatch_count",
            "gpu_ms_median",
            "throughput_median",
            "gbps_median",
            "point_label",
        ]
    ]


def _plot_representative_points(points: pd.DataFrame, output_path: Path) -> Path | None:
    if points.empty:
        return

    x = np.arange(points.shape[0])
    labels = points["point_label"].tolist()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].bar(x, points["gpu_ms_median"], color="#1f77b4")
    axes[0].set_ylabel("gpu_ms")
    axes[0].set_title("Representative points: GPU dispatch time")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, points["throughput_median"] / 1.0e9, color="#2ca02c")
    axes[1].set_ylabel("throughput (Gelem/s)")
    axes[1].set_title("Representative points: throughput")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, points["gbps_median"], color="#ff7f0e")
    axes[2].set_ylabel("GB/s")
    axes[2].set_title("Representative points: effective bandwidth")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment 01 Representative Points (contiguous_write)", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_ratio_summary_table(latest_summary: pd.DataFrame) -> pd.DataFrame:
    if latest_summary.empty:
        return pd.DataFrame()

    write_frame = latest_summary[latest_summary["variant"] == "contiguous_write"][
        ["problem_size", "dispatch_count", "gpu_ms_median", "throughput_median"]
    ].copy()
    noop_frame = latest_summary[latest_summary["variant"] == "noop"][
        ["problem_size", "dispatch_count", "gpu_ms_median", "throughput_median"]
    ].copy()
    if write_frame.empty or noop_frame.empty:
        return pd.DataFrame()

    merged = write_frame.merge(
        noop_frame,
        on=["problem_size", "dispatch_count"],
        suffixes=("_write", "_noop"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["gpu_time_ratio"] = merged["gpu_ms_median_write"] / merged["gpu_ms_median_noop"]
    merged["throughput_ratio"] = merged["throughput_median_write"] / merged["throughput_median_noop"].where(merged["throughput_median_noop"] > 0.0)
    summary = (
        merged.groupby("dispatch_count", as_index=False)
        .agg(
            gpu_time_ratio=("gpu_time_ratio", "median"),
            throughput_ratio=("throughput_ratio", "median"),
        )
        .sort_values("dispatch_count")
        .reset_index(drop=True)
    )

    target_dispatch_counts = [1, 4, 16, 64]
    selection = summary[summary["dispatch_count"].isin(target_dispatch_counts)].copy()
    if not selection.empty:
        return selection
    return summary


def _plot_ratio_summary(summary: pd.DataFrame, output_path: Path) -> Path | None:
    if summary.empty:
        return

    x = np.arange(summary.shape[0])
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2.0, summary["gpu_time_ratio"], width=width, label="GPU Time Ratio")
    if summary["throughput_ratio"].notna().any():
        ax.bar(x + width / 2.0, summary["throughput_ratio"], width=width, label="Throughput Ratio")
    else:
        ax.text(0.02, 0.98, "Throughput ratio undefined: no-op has no useful payload.", va="top", transform=ax.transAxes)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(value)) for value in summary["dispatch_count"]])
    ax.set_xlabel("dispatch_count")
    ax.set_ylabel("ratio")
    ax.set_title("Operation-level median ratio (contiguous_write / noop)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_peak_throughput_table(latest_summary: pd.DataFrame) -> pd.DataFrame:
    if latest_summary.empty:
        return pd.DataFrame()

    records: list[dict] = []
    for variant, frame in latest_summary.groupby("variant", sort=True):
        clean = frame.dropna(subset=["throughput_median"])
        if clean.empty:
            continue
        peak_row = clean.loc[clean["throughput_median"].idxmax()]
        problem_size = int(peak_row["problem_size"])
        records.append(
            {
                "variant": str(variant),
                "problem_size": problem_size,
                "problem_size_label": f"2^{int(round(np.log2(problem_size)))} ({problem_size})",
                "dispatch_count": int(peak_row["dispatch_count"]),
                "throughput_median": float(peak_row["throughput_median"]),
                "throughput_gelem_s": float(peak_row["throughput_median"] / 1.0e9),
                "gbps_median": float(peak_row["gbps_median"]),
                "gpu_ms_median": float(peak_row["gpu_ms_median"]),
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("variant").reset_index(drop=True)


def _plot_peak_throughput(peaks: pd.DataFrame, output_path: Path) -> Path | None:
    if peaks.empty:
        return

    x = np.arange(peaks.shape[0])
    colors = ["#1f77b4" if variant == "contiguous_write" else "#2ca02c" for variant in peaks["variant"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, peaks["throughput_gelem_s"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(peaks["variant"].tolist())
    ax.set_ylabel("throughput (Gelem/s)")
    ax.set_title("Peak throughput by variant")
    ax.grid(axis="y", alpha=0.3)

    for bar, (_, row) in zip(bars, peaks.iterrows()):
        label = f"{row['problem_size_label']}\nd={int(row['dispatch_count'])}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_status_overview_table(latest_rows: pd.DataFrame) -> pd.DataFrame:
    if latest_rows.empty:
        return pd.DataFrame()

    total_rows = int(latest_rows.shape[0])
    correctness_pass_count = int(latest_rows["correctness_pass"].astype(bool).sum())
    correctness_fail_count = total_rows - correctness_pass_count
    correctness_pass_rate = float(correctness_pass_count / total_rows) if total_rows > 0 else 0.0

    gpu_ms_valid_count = int(np.isfinite(pd.to_numeric(latest_rows["gpu_ms"], errors="coerce")).sum())
    end_to_end_valid_count = int(np.isfinite(pd.to_numeric(latest_rows["end_to_end_ms"], errors="coerce")).sum())
    gpu_ms_coverage_rate = float(gpu_ms_valid_count / total_rows) if total_rows > 0 else 0.0
    end_to_end_coverage_rate = float(end_to_end_valid_count / total_rows) if total_rows > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "total_rows": total_rows,
                "correctness_pass_count": correctness_pass_count,
                "correctness_fail_count": correctness_fail_count,
                "correctness_pass_rate": correctness_pass_rate,
                "gpu_ms_valid_count": gpu_ms_valid_count,
                "gpu_ms_coverage_rate": gpu_ms_coverage_rate,
                "end_to_end_ms_valid_count": end_to_end_valid_count,
                "end_to_end_ms_coverage_rate": end_to_end_coverage_rate,
            }
        ]
    )


def _plot_status_overview(status: pd.DataFrame, output_path: Path) -> Path | None:
    if status.empty:
        return

    row = status.iloc[0]
    pass_count = int(row["correctness_pass_count"])
    fail_count = int(row["correctness_fail_count"])
    pass_rate_percent = float(row["correctness_pass_rate"] * 100.0)
    gpu_cov_percent = float(row["gpu_ms_coverage_rate"] * 100.0)
    end_cov_percent = float(row["end_to_end_ms_coverage_rate"] * 100.0)
    total_rows = int(row["total_rows"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(["correctness"], [pass_count], color="#2ca02c", label="pass")
    axes[0].barh(["correctness"], [fail_count], left=[pass_count], color="#d62728", label="fail")
    axes[0].set_title(f"Correctness: {pass_rate_percent:.1f}% pass ({pass_count}/{total_rows})")
    axes[0].set_xlabel("measured rows")
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].legend(loc="lower right")

    coverage_labels = ["gpu_ms", "end_to_end_ms"]
    coverage_values = [gpu_cov_percent, end_cov_percent]
    coverage_colors = ["#1f77b4", "#ff7f0e"]
    bars = axes[1].bar(coverage_labels, coverage_values, color=coverage_colors)
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("coverage (%)")
    axes[1].set_title("Timing Field Coverage")
    axes[1].grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, coverage_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f"{value:.1f}%", ha="center", va="bottom")

    fig.suptitle("Experiment 01 Run Status Overview", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _build_test_setup_table(latest_rows: pd.DataFrame, latest_run_info: pd.Series) -> pd.DataFrame:
    if latest_rows.empty:
        return pd.DataFrame()

    problem_size_min = int(latest_rows["problem_size"].min())
    problem_size_max = int(latest_rows["problem_size"].max())
    problem_size_min_pow = int(round(np.log2(problem_size_min)))
    problem_size_max_pow = int(round(np.log2(problem_size_max)))

    dispatch_counts = sorted(int(value) for value in latest_rows["dispatch_count"].unique().tolist())
    variants = sorted(str(value) for value in latest_rows["variant"].unique().tolist())

    warmup_iterations = int(latest_run_info.get("warmup_iterations", 0))
    timed_iterations = int(latest_run_info.get("timed_iterations", 0))
    validation_enabled = bool(latest_run_info.get("validation_enabled", False))

    records = [
        {"field": "GPU", "value": str(latest_run_info.get("gpu_name", "unknown"))},
        {"field": "Vulkan API", "value": str(latest_run_info.get("vulkan_api_version", "unknown"))},
        {"field": "Validation", "value": "enabled" if validation_enabled else "disabled"},
        {"field": "Warmup", "value": str(warmup_iterations)},
        {"field": "Timed Iterations", "value": str(timed_iterations)},
        {
            "field": "Problem Size Sweep",
            "value": f"2^{problem_size_min_pow} .. 2^{problem_size_max_pow} ({problem_size_min} .. {problem_size_max})",
        },
        {"field": "Dispatch Count Sweep", "value": "{" + ", ".join(str(value) for value in dispatch_counts) + "}"},
        {"field": "Variants", "value": ", ".join(variants)},
    ]
    return pd.DataFrame(records)


def _plot_test_setup_overview(test_setup: pd.DataFrame, output_path: Path) -> Path | None:
    if test_setup.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_axis_off()

    ax.text(
        0.02,
        0.95,
        "Experiment 01 Test Setup (from latest results log)",
        fontsize=16,
        fontweight="bold",
        va="top",
        transform=ax.transAxes,
    )

    start_y = 0.84
    row_step = 0.10
    for index, (_, row) in enumerate(test_setup.iterrows()):
        y = start_y - (index * row_step)
        ax.text(0.03, y, f"{row['field']}:", fontsize=12, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0.28, y, str(row["value"]), fontsize=12, va="top", transform=ax.transAxes)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _device_configuration_groups(summary: pd.DataFrame):
    """Keep every curve and per-device chart within one comparable run configuration."""
    for _, frame in summary.groupby(DEVICE_COLUMNS + RUN_CONFIGURATION_COLUMNS, sort=True):
        row = frame.iloc[0]
        validation = "enabled" if row["validation_enabled"] else "disabled"
        settings_label = (
            f"validation {validation}, warmup {row['warmup_iterations']}, timed {row['timed_iterations']}"
        )
        label = f"{row['device_id']} | {settings_label}"
        slug = row["device_slug"]
        device_configurations = summary[summary["device_id"] == row["device_id"]][RUN_CONFIGURATION_COLUMNS]
        if len(device_configurations.drop_duplicates()) > 1:
            slug += "_" + _slugify(f"{row['measurement_revision']}_{settings_label}")
        yield label, slug, frame


def _build_chart_index(device_summary: pd.DataFrame) -> pd.DataFrame:
    if device_summary.empty:
        return pd.DataFrame(columns=DEVICE_COLUMNS + RUN_CONFIGURATION_COLUMNS + ["chart"])

    records: list[dict] = []
    for _, chart_slug, frame in _device_configuration_groups(device_summary):
        record = frame.iloc[0][DEVICE_COLUMNS + RUN_CONFIGURATION_COLUMNS].to_dict()
        record["chart"] = f"results/charts/dispatch_basics_{chart_slug}_time_throughput.png"
        records.append(record)
    return pd.DataFrame(records).sort_values(DEVICE_COLUMNS + RUN_CONFIGURATION_COLUMNS).reset_index(drop=True)


def main() -> None:
    generated_tables: list[Path] = []
    generated_charts: list[Path | None] = []
    parser = argparse.ArgumentParser(description="Analyze Experiment 01 dispatch basics runs across devices.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing collected run JSON files (default: experiments/01_dispatch_basics/runs).",
    )
    parser.add_argument(
        "--skip-current",
        action="store_true",
        help="Do not include results/tables/benchmark_results.json as an implicit run.",
    )
    parser.add_argument(
        "--measurement-revision",
        help="Analyze one revision only (default: current run, otherwise newest archive); use legacy-unversioned for old runs.",
    )
    args = parser.parse_args()

    run_files = _discover_run_files(args.runs_dir, include_current=not args.skip_current, revision=args.measurement_revision)
    all_rows, run_index = _load_all_runs(run_files)
    if all_rows.empty or run_index.empty:
        raise FileNotFoundError(
            "No Experiment 01 rows found. Add runs under experiments/01_dispatch_basics/runs or generate "
            "results/tables/benchmark_results.json first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    run_index_export = run_index.drop(columns=["exported_at_dt"]).copy()
    generated_tables.append(_write_table(run_index_export, TABLES_DIR / "dispatch_basics_runs_index.csv"))

    run_summary = _build_run_summary(all_rows)
    generated_tables.append(_write_table(run_summary, TABLES_DIR / "dispatch_basics_multi_run_summary.csv"))

    latest_run_id = preferred_run_id(run_index, CURRENT_RUN_JSON if not args.skip_current else None, ROOT)
    latest_run_info = run_index[run_index["run_id"] == latest_run_id].iloc[0]
    latest_rows = all_rows[all_rows["run_id"] == latest_run_id].copy()
    latest_summary = run_summary[run_summary["run_id"] == latest_run_id].copy()
    latest_summary_simple = latest_summary[
        [
            "variant",
            "problem_size",
            "dispatch_count",
            "gpu_ms_median",
            "gpu_ms_p95",
            "end_to_end_ms_median",
            "throughput_median",
            "throughput_p95",
            "gbps_median",
            "correctness_pass_rate",
            "sample_count",
        ]
    ].sort_values(["variant", "problem_size", "dispatch_count"])
    generated_tables.append(_write_table(latest_summary_simple, TABLES_DIR / "dispatch_basics_summary.csv"))

    latest_best_dispatch = _best_dispatch_table(latest_summary_simple, ["variant", "problem_size"])
    generated_tables.append(_write_table(latest_best_dispatch, TABLES_DIR / "dispatch_basics_best_dispatch.csv"))

    gpu_ms_pivot = latest_summary_simple.pivot_table(
        index="problem_size",
        columns=["variant", "dispatch_count"],
        values="gpu_ms_median",
    ).sort_index()
    throughput_pivot = latest_summary_simple.pivot_table(
        index="problem_size",
        columns=["variant", "dispatch_count"],
        values="throughput_median",
    ).sort_index()
    gpu_ms_pivot.to_csv(TABLES_DIR / "dispatch_basics_gpu_ms_pivot.csv")
    generated_tables.append(TABLES_DIR / "dispatch_basics_gpu_ms_pivot.csv")
    throughput_pivot.to_csv(TABLES_DIR / "dispatch_basics_throughput_pivot.csv")
    generated_tables.append(TABLES_DIR / "dispatch_basics_throughput_pivot.csv")

    device_summary = _build_device_summary(run_summary)
    generated_tables.append(_write_table(device_summary, TABLES_DIR / "dispatch_basics_device_summary.csv"))

    device_best_dispatch = _best_dispatch_table(
        device_summary,
        DEVICE_COLUMNS + RUN_CONFIGURATION_COLUMNS + ["variant", "problem_size"],
    )
    generated_tables.append(_write_table(device_best_dispatch, TABLES_DIR / "dispatch_basics_best_dispatch_by_device.csv"))

    operation_ratios = _build_operation_ratio_table(device_summary)
    if not operation_ratios.empty:
        generated_tables.append(_write_table(operation_ratios, TABLES_DIR / "dispatch_basics_operation_ratio_by_device.csv"))

    chart_index = _build_chart_index(device_summary)
    generated_tables.append(_write_table(chart_index, TABLES_DIR / "dispatch_basics_device_chart_index.csv"))

    representative_points = _build_representative_points_table(latest_summary_simple)
    if not representative_points.empty:
        generated_tables.append(_write_table(representative_points, TABLES_DIR / "dispatch_basics_representative_points.csv"))
        generated_charts.append(_plot_representative_points(representative_points, CHARTS_DIR / "dispatch_basics_representative_points.png"))

    ratio_summary = _build_ratio_summary_table(latest_summary_simple)
    if not ratio_summary.empty:
        generated_tables.append(_write_table(ratio_summary, TABLES_DIR / "dispatch_basics_ratio_summary.csv"))
        generated_charts.append(_plot_ratio_summary(ratio_summary, CHARTS_DIR / "dispatch_basics_ratio_summary.png"))

    peak_throughput = _build_peak_throughput_table(latest_summary_simple)
    if not peak_throughput.empty:
        generated_tables.append(_write_table(peak_throughput, TABLES_DIR / "dispatch_basics_peak_throughput.csv"))
        generated_charts.append(_plot_peak_throughput(peak_throughput, CHARTS_DIR / "dispatch_basics_peak_throughput.png"))

    status_overview = _build_status_overview_table(latest_rows)
    if not status_overview.empty:
        generated_tables.append(_write_table(status_overview, TABLES_DIR / "dispatch_basics_status_overview.csv"))
        generated_charts.append(_plot_status_overview(status_overview, CHARTS_DIR / "dispatch_basics_status_overview.png"))

    test_setup = _build_test_setup_table(latest_rows, latest_run_info)
    if not test_setup.empty:
        generated_tables.append(_write_table(test_setup, TABLES_DIR / "dispatch_basics_test_setup.csv"))
        generated_charts.append(_plot_test_setup_overview(test_setup, CHARTS_DIR / "dispatch_basics_test_setup.png"))

    latest_device_label = latest_summary["device_id"].iloc[0]
    generated_charts.append(_plot_time_throughput(
        latest_summary_simple,
        CHARTS_DIR / "dispatch_basics_time_throughput.png",
        f"Experiment 01 Dispatch Basics ({latest_device_label})",
    ))

    for device_label, chart_slug, frame in _device_configuration_groups(device_summary):
        generated_charts.append(_plot_time_throughput(
            frame,
            CHARTS_DIR / f"dispatch_basics_{chart_slug}_time_throughput.png",
            f"Experiment 01 Dispatch Basics ({device_label})",
        ))

    generated_charts.append(_plot_cross_device_comparison(device_summary, CHARTS_DIR / "dispatch_basics_cross_device_comparison.png"))
    generated_charts.append(_plot_operation_ratio_by_device(operation_ratios, CHARTS_DIR / "dispatch_basics_operation_ratio_by_device.png"))

    write_analysis_provenance(TABLES_DIR, run_files, EXPERIMENT_ID, derived_tables=generated_tables, derived_charts=generated_charts, analysis_tools=[Path(__file__)])
    print(f"Processed runs: {len(run_index)}")
    print(f"Latest run: {latest_run_id}")
    print(f"Devices in dataset: {run_index['device_id'].nunique()}")
    print(f"Wrote tables to: {TABLES_DIR}")
    print(f"Wrote charts to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
