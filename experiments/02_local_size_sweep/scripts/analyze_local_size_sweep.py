#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_ID = "02_local_size_sweep"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"
CURRENT_RUN_JSON = TABLES_DIR / "benchmark_results.json"

LOCAL_SIZE_RE = re.compile(r"_ls(\d+)$")


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
        exponent = int(round(np.log2(value)))
        labels.append(f"2^{exponent}")
    return labels


def _extract_local_size(variant: str) -> int | None:
    match = LOCAL_SIZE_RE.search(variant)
    if match is None:
        return None
    return int(match.group(1))


def _base_variant(variant: str) -> str:
    return LOCAL_SIZE_RE.sub("", variant)


def _discover_run_files(runs_dir: Path, include_current: bool) -> list[Path]:
    paths = sorted(runs_dir.rglob("*.json")) if runs_dir.exists() else []
    if include_current and CURRENT_RUN_JSON.exists():
        current_resolved = CURRENT_RUN_JSON.resolve()
        if all(path.resolve() != current_resolved for path in paths):
            paths.append(CURRENT_RUN_JSON)
    return paths


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
        "run_signature": f"{gpu_name}|{driver_version}|{vulkan_api_version}|"
        f"{exported_at_utc or exported_at_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}|{int(frame.shape[0])}",
        "row_count": int(frame.shape[0]),
        "correctness_pass_rate": float(frame["correctness_pass"].mean()),
        "problem_size_min": int(frame["problem_size"].min()),
        "problem_size_max": int(frame["problem_size"].max()),
        "local_sizes": ",".join(str(int(v)) for v in sorted(frame["local_size_x"].unique())),
        "dispatch_counts": ",".join(str(int(v)) for v in sorted(frame["dispatch_count"].unique())),
    }


def _load_local_size_rows(path: Path) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] Skipping unreadable JSON: {path} ({exc})")
        return None, None

    rows = payload.get("rows", [])
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
    frame["local_size_x"] = frame["variant"].apply(_extract_local_size)
    frame = frame.dropna(subset=["local_size_x"]).copy()
    if frame.empty:
        return None, None
    frame["local_size_x"] = frame["local_size_x"].astype(int)
    frame["base_variant"] = frame["variant"].apply(_base_variant)

    if "notes" in frame.columns:
        frame["notes"] = frame["notes"].fillna("").astype(str)
    else:
        frame["notes"] = ""

    if "correctness_pass" in frame.columns:
        frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
    else:
        frame["correctness_pass"] = False

    run_record = _build_run_record(path, payload, frame)
    for key in [
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
    seen_signatures: set[str] = set()

    for path in paths:
        frame, run_record = _load_local_size_rows(path)
        if frame is None or run_record is None:
            continue

        run_signature = run_record["run_signature"]
        if run_signature in seen_signatures:
            print(f"[info] Skipping duplicate run payload: {_relative_run_id(path)}")
            continue
        seen_signatures.add(run_signature)

        frames.append(frame)
        run_records.append(run_record)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    all_rows = pd.concat(frames, ignore_index=True)
    run_index = pd.DataFrame(run_records).sort_values(["exported_at_dt", "run_id"]).reset_index(drop=True)
    return all_rows, run_index


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def _build_run_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "run_id",
        "run_file",
        "gpu_name",
        "driver_version",
        "vulkan_api_version",
        "validation_enabled",
        "exported_at_utc",
        "device_id",
        "device_slug",
        "base_variant",
        "local_size_x",
        "problem_size",
        "dispatch_count",
    ]
    return (
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
        .sort_values(
            ["exported_at_utc", "run_id", "base_variant", "problem_size", "dispatch_count", "local_size_x"]
        )
        .reset_index(drop=True)
    )


def _build_best_local_size_table(summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for group_key, group in summary.groupby(["base_variant", "problem_size", "dispatch_count"], sort=True):
        clean_group = group.dropna(subset=["gpu_ms_median", "throughput_median"])
        if clean_group.empty:
            continue

        best_latency = clean_group.loc[clean_group["gpu_ms_median"].idxmin()]
        best_throughput = clean_group.loc[clean_group["throughput_median"].idxmax()]
        records.append(
            {
                "base_variant": str(group_key[0]),
                "problem_size": int(group_key[1]),
                "dispatch_count": int(group_key[2]),
                "best_local_size_for_latency": int(best_latency["local_size_x"]),
                "best_gpu_ms": float(best_latency["gpu_ms_median"]),
                "best_local_size_for_throughput": int(best_throughput["local_size_x"]),
                "best_throughput": float(best_throughput["throughput_median"]),
            }
        )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["base_variant", "problem_size", "dispatch_count"]).reset_index(drop=True)


def _build_speedup_vs_ls64_table(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["local_size_x"] == 64][
        ["base_variant", "problem_size", "dispatch_count", "gpu_ms_median", "throughput_median"]
    ].copy()
    if baseline.empty:
        return pd.DataFrame()

    merged = summary.merge(
        baseline,
        on=["base_variant", "problem_size", "dispatch_count"],
        suffixes=("", "_ls64"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["latency_speedup_vs_ls64"] = merged["gpu_ms_median_ls64"] / merged["gpu_ms_median"]
    merged["throughput_speedup_vs_ls64"] = merged["throughput_median"] / merged["throughput_median_ls64"]
    columns = [
        "base_variant",
        "problem_size",
        "dispatch_count",
        "local_size_x",
        "gpu_ms_median",
        "gpu_ms_median_ls64",
        "latency_speedup_vs_ls64",
        "throughput_median",
        "throughput_median_ls64",
        "throughput_speedup_vs_ls64",
    ]
    return (
        merged[columns]
        .sort_values(["base_variant", "problem_size", "dispatch_count", "local_size_x"])
        .reset_index(drop=True)
    )


def _build_local_size_ranking_table(speedup_vs_ls64: pd.DataFrame) -> pd.DataFrame:
    if speedup_vs_ls64.empty:
        return pd.DataFrame()

    target_variant = "contiguous_write"
    ranking_input = speedup_vs_ls64[speedup_vs_ls64["base_variant"] == target_variant].copy()
    if ranking_input.empty:
        target_variant = sorted(speedup_vs_ls64["base_variant"].unique().tolist())[0]
        ranking_input = speedup_vs_ls64[speedup_vs_ls64["base_variant"] == target_variant].copy()

    ranking_input = ranking_input[ranking_input["latency_speedup_vs_ls64"] > 0.0]
    if ranking_input.empty:
        return pd.DataFrame()

    ranking = (
        ranking_input.groupby("local_size_x", as_index=False)
        .agg(
            geometric_mean_speedup=("latency_speedup_vs_ls64", lambda s: float(np.exp(np.mean(np.log(s))))),
            median_speedup=("latency_speedup_vs_ls64", "median"),
            sample_count=("latency_speedup_vs_ls64", "count"),
        )
        .sort_values("geometric_mean_speedup", ascending=False)
        .reset_index(drop=True)
    )
    ranking["base_variant"] = target_variant
    return ranking[
        ["base_variant", "local_size_x", "geometric_mean_speedup", "median_speedup", "sample_count"]
    ].copy()


def _build_operation_ratio_table(summary: pd.DataFrame) -> pd.DataFrame:
    variants = set(summary["base_variant"].unique().tolist())
    if "contiguous_write" not in variants or "noop" not in variants:
        return pd.DataFrame()

    write_frame = summary[summary["base_variant"] == "contiguous_write"].copy()
    noop_frame = summary[summary["base_variant"] == "noop"].copy()
    join_columns = ["problem_size", "dispatch_count", "local_size_x"]
    merged = write_frame.merge(noop_frame, on=join_columns, suffixes=("_write", "_noop"), how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["gpu_ms_ratio_write_over_noop"] = merged["gpu_ms_median_write"] / merged["gpu_ms_median_noop"]
    merged["throughput_ratio_write_over_noop"] = (
        merged["throughput_median_write"] / merged["throughput_median_noop"]
    )
    return (
        merged[
            join_columns
            + [
                "gpu_ms_ratio_write_over_noop",
                "throughput_ratio_write_over_noop",
            ]
        ]
        .sort_values(["problem_size", "dispatch_count", "local_size_x"])
        .reset_index(drop=True)
    )


def _build_operation_ratio_summary_table(operation_ratio: pd.DataFrame) -> pd.DataFrame:
    if operation_ratio.empty:
        return pd.DataFrame()

    return (
        operation_ratio.groupby("local_size_x", as_index=False)
        .agg(
            gpu_ms_ratio_write_over_noop=("gpu_ms_ratio_write_over_noop", "median"),
            throughput_ratio_write_over_noop=("throughput_ratio_write_over_noop", "median"),
        )
        .sort_values("local_size_x")
        .reset_index(drop=True)
    )


def _plot_metric_vs_local_size(
    summary: pd.DataFrame, metric_column: str, y_label: str, output_path: Path, title: str
) -> None:
    if summary.empty:
        return

    variants = sorted(summary["base_variant"].unique().tolist())
    local_sizes = np.array(sorted(summary["local_size_x"].unique().tolist()), dtype=float)
    if len(local_sizes) == 0:
        return

    fig, axes = plt.subplots(len(variants), 1, figsize=(12, 4.5 * len(variants)), sharex=True)
    if len(variants) == 1:
        axes = np.array([axes])

    for row_index, variant in enumerate(variants):
        subset = summary[summary["base_variant"] == variant].copy()
        for (dispatch_count, problem_size), curve in subset.groupby(["dispatch_count", "problem_size"], sort=True):
            curve = curve.sort_values("local_size_x")
            if curve.empty:
                continue
            label = f"d={int(dispatch_count)}, size=2^{int(round(np.log2(problem_size)))}"
            axes[row_index].plot(curve["local_size_x"], curve[metric_column], marker="o", label=label)

        axes[row_index].set_title(f"{variant}: {y_label} vs local_size_x")
        axes[row_index].set_ylabel(y_label)
        axes[row_index].set_xscale("log", base=2)
        axes[row_index].grid(True, alpha=0.3)
        axes[row_index].legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel("local_size_x")
    axes[-1].set_xticks(local_sizes)
    axes[-1].set_xticklabels([str(int(v)) for v in local_sizes])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_operation_ratio_summary(operation_ratio_summary: pd.DataFrame, output_path: Path) -> None:
    if operation_ratio_summary.empty:
        return

    x = np.arange(operation_ratio_summary.shape[0])
    width = 0.36
    labels = [str(int(value)) for value in operation_ratio_summary["local_size_x"].tolist()]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        x - width / 2.0,
        operation_ratio_summary["gpu_ms_ratio_write_over_noop"],
        width=width,
        label="GPU Time Ratio",
        color="#1f77b4",
    )
    ax.bar(
        x + width / 2.0,
        operation_ratio_summary["throughput_ratio_write_over_noop"],
        width=width,
        label="Throughput Ratio",
        color="#ff7f0e",
    )
    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("local_size_x")
    ax.set_ylabel("median ratio")
    ax.set_title("Operation-Level Ratio (contiguous_write / noop)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_speedup_vs_ls64(speedup_vs_ls64: pd.DataFrame, output_path: Path) -> None:
    if speedup_vs_ls64.empty:
        return

    aggregate = (
        speedup_vs_ls64.groupby(["base_variant", "local_size_x"], as_index=False)
        .agg(latency_speedup_vs_ls64=("latency_speedup_vs_ls64", "median"))
        .sort_values(["base_variant", "local_size_x"])
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for variant, frame in aggregate.groupby("base_variant"):
        ax.plot(
            frame["local_size_x"],
            frame["latency_speedup_vs_ls64"],
            marker="o",
            label=str(variant),
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("local_size_x")
    ax.set_ylabel("latency speedup vs local_size_x=64 (median)")
    ax.set_title("Local Size Sweep: Speedup vs local_size_x=64")
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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


def _plot_status_overview(status: pd.DataFrame, output_path: Path) -> None:
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
    bars = axes[1].bar(coverage_labels, coverage_values, color=["#1f77b4", "#ff7f0e"])
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("coverage (%)")
    axes[1].set_title("Timing Field Coverage")
    axes[1].grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, coverage_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f"{value:.1f}%", ha="center", va="bottom")

    fig.suptitle("Experiment 02 Run Status Overview", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_test_setup_table(latest_rows: pd.DataFrame, latest_run_info: pd.Series) -> pd.DataFrame:
    if latest_rows.empty:
        return pd.DataFrame()

    problem_size_values = np.array(sorted(latest_rows["problem_size"].unique().tolist()), dtype=float)
    local_size_values = sorted(int(v) for v in latest_rows["local_size_x"].unique().tolist())
    dispatch_counts = sorted(int(v) for v in latest_rows["dispatch_count"].unique().tolist())
    variants = sorted(str(v) for v in latest_rows["base_variant"].unique().tolist())

    problem_size_min = int(problem_size_values.min())
    problem_size_max = int(problem_size_values.max())
    problem_size_min_pow = int(round(np.log2(problem_size_min)))
    problem_size_max_pow = int(round(np.log2(problem_size_max)))

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
        {"field": "Local Size Sweep", "value": "{" + ", ".join(str(v) for v in local_size_values) + "}"},
        {"field": "Dispatch Count Sweep", "value": "{" + ", ".join(str(v) for v in dispatch_counts) + "}"},
        {"field": "Variants", "value": ", ".join(variants)},
    ]
    return pd.DataFrame(records)


def _plot_test_setup_overview(test_setup: pd.DataFrame, output_path: Path) -> None:
    if test_setup.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_axis_off()

    ax.text(
        0.02,
        0.95,
        "Experiment 02 Test Setup (from latest results log)",
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
        ax.text(0.30, y, str(row["value"]), fontsize=12, va="top", transform=ax.transAxes)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 02 local size sweep runs across devices.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing collected run JSON files (default: experiments/02_local_size_sweep/runs).",
    )
    parser.add_argument(
        "--skip-current",
        action="store_true",
        help="Do not include results/tables/benchmark_results.json as an implicit run.",
    )
    args = parser.parse_args()

    run_files = _discover_run_files(args.runs_dir, include_current=not args.skip_current)
    all_rows, run_index = _load_all_runs(run_files)
    if all_rows.empty or run_index.empty:
        raise FileNotFoundError(
            "No Experiment 02 rows found. Add runs under experiments/02_local_size_sweep/runs or generate "
            "results/tables/benchmark_results.json first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    run_index_export = run_index.drop(columns=["exported_at_dt", "run_signature"]).copy()
    _write_table(run_index_export, TABLES_DIR / "local_size_sweep_runs_index.csv")

    run_summary = _build_run_summary(all_rows)
    _write_table(run_summary, TABLES_DIR / "local_size_sweep_multi_run_summary.csv")

    latest_run_id = run_index.sort_values(["exported_at_dt", "run_id"]).iloc[-1]["run_id"]
    latest_run_info = run_index[run_index["run_id"] == latest_run_id].iloc[0]
    latest_rows = all_rows[all_rows["run_id"] == latest_run_id].copy()
    latest_summary = run_summary[run_summary["run_id"] == latest_run_id].copy()
    latest_summary_simple = latest_summary[
        [
            "base_variant",
            "local_size_x",
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
    ].sort_values(["base_variant", "problem_size", "dispatch_count", "local_size_x"])
    _write_table(latest_summary_simple, TABLES_DIR / "local_size_sweep_summary.csv")

    best_local_size = _build_best_local_size_table(latest_summary_simple)
    if not best_local_size.empty:
        _write_table(best_local_size, TABLES_DIR / "local_size_sweep_best_local_size.csv")

    gpu_ms_pivot = latest_summary_simple.pivot_table(
        index="problem_size",
        columns=["base_variant", "dispatch_count", "local_size_x"],
        values="gpu_ms_median",
    ).sort_index()
    throughput_pivot = latest_summary_simple.pivot_table(
        index="problem_size",
        columns=["base_variant", "dispatch_count", "local_size_x"],
        values="throughput_median",
    ).sort_index()
    gpu_ms_pivot.to_csv(TABLES_DIR / "local_size_sweep_gpu_ms_pivot.csv")
    throughput_pivot.to_csv(TABLES_DIR / "local_size_sweep_throughput_pivot.csv")

    speedup_vs_ls64 = _build_speedup_vs_ls64_table(latest_summary_simple)
    if not speedup_vs_ls64.empty:
        _write_table(speedup_vs_ls64, TABLES_DIR / "local_size_sweep_speedup_vs_ls64.csv")

    local_size_ranking = _build_local_size_ranking_table(speedup_vs_ls64)
    if not local_size_ranking.empty:
        _write_table(local_size_ranking, TABLES_DIR / "local_size_sweep_local_size_ranking.csv")

    operation_ratio = _build_operation_ratio_table(latest_summary_simple)
    if not operation_ratio.empty:
        _write_table(operation_ratio, TABLES_DIR / "local_size_sweep_operation_ratio.csv")
    operation_ratio_summary = _build_operation_ratio_summary_table(operation_ratio)
    if not operation_ratio_summary.empty:
        _write_table(operation_ratio_summary, TABLES_DIR / "local_size_sweep_operation_ratio_summary.csv")

    status_overview = _build_status_overview_table(latest_rows)
    if not status_overview.empty:
        _write_table(status_overview, TABLES_DIR / "local_size_sweep_status_overview.csv")

    test_setup = _build_test_setup_table(latest_rows, latest_run_info)
    if not test_setup.empty:
        _write_table(test_setup, TABLES_DIR / "local_size_sweep_test_setup.csv")

    _plot_metric_vs_local_size(
        latest_summary_simple,
        metric_column="gpu_ms_median",
        y_label="gpu_ms (median)",
        output_path=CHARTS_DIR / "local_size_sweep_gpu_ms_vs_local_size.png",
        title="Experiment 02 Local Size Sweep: GPU Time",
    )
    _plot_metric_vs_local_size(
        latest_summary_simple,
        metric_column="throughput_median",
        y_label="throughput (elements/s, median)",
        output_path=CHARTS_DIR / "local_size_sweep_throughput_vs_local_size.png",
        title="Experiment 02 Local Size Sweep: Throughput",
    )
    _plot_speedup_vs_ls64(speedup_vs_ls64, CHARTS_DIR / "local_size_sweep_speedup_vs_ls64.png")
    _plot_operation_ratio_summary(operation_ratio_summary, CHARTS_DIR / "local_size_sweep_operation_ratio_summary.png")
    _plot_status_overview(status_overview, CHARTS_DIR / "local_size_sweep_status_overview.png")
    _plot_test_setup_overview(test_setup, CHARTS_DIR / "local_size_sweep_test_setup.png")

    print(f"Processed runs: {len(run_index)}")
    print(f"Latest run: {latest_run_id}")
    print(f"Devices in dataset: {run_index['device_id'].nunique()}")
    print(f"Wrote tables to: {TABLES_DIR}")
    print(f"Wrote charts to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
