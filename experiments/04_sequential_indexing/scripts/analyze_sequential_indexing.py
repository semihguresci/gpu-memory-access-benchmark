#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_ID = "04_sequential_indexing"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"
CURRENT_RUN_JSON = TABLES_DIR / "benchmark_results.json"


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


def _load_rows(path: Path) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
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
    frame = frame.dropna(subset=["problem_size", "dispatch_count", "gpu_ms", "gbps"])
    if frame.empty:
        return None, None

    frame["problem_size"] = frame["problem_size"].astype(int)
    frame["dispatch_count"] = frame["dispatch_count"].astype(int)
    frame["iteration"] = frame["iteration"].fillna(0).astype(int)
    frame["variant"] = frame["variant"].fillna("unknown_variant").astype(str)
    if "correctness_pass" in frame.columns:
        frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
    else:
        frame["correctness_pass"] = False

    metadata = payload.get("metadata", {})
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    driver_version = str(metadata.get("driver_version", "unknown_driver"))
    vulkan_api_version = str(metadata.get("vulkan_api_version", "unknown_api"))
    validation_enabled = bool(metadata.get("validation_enabled", False))
    warmup_iterations = int(metadata.get("warmup_iterations", 0))
    timed_iterations = int(metadata.get("timed_iterations", 0))

    exported_at_utc = str(metadata.get("exported_at_utc", ""))
    exported_at_dt = _parse_iso8601(exported_at_utc)
    if exported_at_dt is None:
        exported_at_dt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    run_id = _relative_run_id(path)
    device_id = f"{gpu_name} | drv {driver_version} | vk {vulkan_api_version}"
    frame["run_id"] = run_id
    frame["gpu_name"] = gpu_name
    frame["driver_version"] = driver_version
    frame["vulkan_api_version"] = vulkan_api_version
    frame["validation_enabled"] = validation_enabled
    frame["exported_at_utc"] = exported_at_utc or exported_at_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["device_id"] = device_id

    record = {
        "run_id": run_id,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "vulkan_api_version": vulkan_api_version,
        "validation_enabled": validation_enabled,
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "exported_at_utc": exported_at_utc or exported_at_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exported_at_dt": exported_at_dt,
        "device_id": device_id,
        "row_count": int(frame.shape[0]),
        "correctness_pass_rate": float(frame["correctness_pass"].mean()),
    }
    return frame, record


def _load_all_runs(paths: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    records: list[dict] = []

    for path in paths:
        frame, record = _load_rows(path)
        if frame is None or record is None:
            continue
        frames.append(frame)
        records.append(record)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    all_rows = pd.concat(frames, ignore_index=True)
    run_index = pd.DataFrame(records).sort_values(["exported_at_dt", "run_id"]).reset_index(drop=True)
    return all_rows, run_index


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "run_id",
        "gpu_name",
        "driver_version",
        "vulkan_api_version",
        "validation_enabled",
        "exported_at_utc",
        "device_id",
        "variant",
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
            gbps_median=("gbps", "median"),
            correctness_pass_rate=("correctness_pass", "mean"),
            sample_count=("gpu_ms", "count"),
        )
        .sort_values(["exported_at_utc", "run_id", "variant", "problem_size", "dispatch_count"])
        .reset_index(drop=True)
    )


def _build_status_overview(latest_rows: pd.DataFrame) -> pd.DataFrame:
    if latest_rows.empty:
        return pd.DataFrame()

    total_rows = int(latest_rows.shape[0])
    pass_rows = int(latest_rows["correctness_pass"].astype(bool).sum())
    fail_rows = total_rows - pass_rows
    pass_rate = float(pass_rows / total_rows) if total_rows > 0 else 0.0
    return pd.DataFrame(
        [
            {
                "total_rows": total_rows,
                "correctness_pass_count": pass_rows,
                "correctness_fail_count": fail_rows,
                "correctness_pass_rate": pass_rate,
            }
        ]
    )


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def _format_problem_size_ticks(values: np.ndarray) -> list[str]:
    labels: list[str] = []
    for value in values:
        exponent = int(round(np.log2(value)))
        labels.append(f"2^{exponent}")
    return labels


def _plot_metric(summary: pd.DataFrame, metric_column: str, y_label: str, title: str, output_path: Path) -> None:
    if summary.empty:
        return

    variants = sorted(summary["variant"].unique().tolist())
    dispatch_counts = sorted(summary["dispatch_count"].unique().tolist())
    if len(dispatch_counts) == 0:
        return

    representative_dispatch_count = dispatch_counts[0]
    filtered = summary[summary["dispatch_count"] == representative_dispatch_count].copy()
    if filtered.empty:
        return

    problem_sizes = np.array(sorted(filtered["problem_size"].unique().tolist()), dtype=float)
    if len(problem_sizes) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for variant in variants:
        curve = filtered[filtered["variant"] == variant].sort_values("problem_size")
        if curve.empty:
            continue
        ax.plot(curve["problem_size"], curve[metric_column], marker="o", label=variant)

    ax.set_xscale("log", base=2)
    ax.set_xticks(problem_sizes)
    ax.set_xticklabels(_format_problem_size_ticks(problem_sizes), rotation=30)
    ax.set_xlabel("problem_size (elements)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} (dispatch_count={representative_dispatch_count})")
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 04 sequential indexing runs.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing collected run JSON files (default: experiments/04_sequential_indexing/runs).",
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
            "No Experiment 04 rows found. Add runs under experiments/04_sequential_indexing/runs or generate "
            "results/tables/benchmark_results.json first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    run_index_export = run_index.drop(columns=["exported_at_dt"]).copy()
    _write_table(run_index_export, TABLES_DIR / "sequential_indexing_runs_index.csv")

    summary = _build_summary(all_rows)
    _write_table(summary, TABLES_DIR / "sequential_indexing_multi_run_summary.csv")

    latest_run_id = run_index.sort_values(["exported_at_dt", "run_id"]).iloc[-1]["run_id"]
    latest_rows = all_rows[all_rows["run_id"] == latest_run_id].copy()
    latest_summary = summary[summary["run_id"] == latest_run_id].copy()
    latest_summary_simple = latest_summary[
        [
            "variant",
            "problem_size",
            "dispatch_count",
            "gpu_ms_median",
            "gpu_ms_p95",
            "end_to_end_ms_median",
            "throughput_median",
            "gbps_median",
            "correctness_pass_rate",
            "sample_count",
        ]
    ].sort_values(["variant", "problem_size", "dispatch_count"])
    _write_table(latest_summary_simple, TABLES_DIR / "sequential_indexing_summary.csv")

    status = _build_status_overview(latest_rows)
    if not status.empty:
        _write_table(status, TABLES_DIR / "sequential_indexing_status_overview.csv")

    _plot_metric(
        latest_summary_simple,
        metric_column="gbps_median",
        y_label="GB/s (median)",
        title="Experiment 04 Sequential Indexing: Effective Bandwidth",
        output_path=CHARTS_DIR / "sequential_indexing_gbps_vs_size.png",
    )
    _plot_metric(
        latest_summary_simple,
        metric_column="gpu_ms_median",
        y_label="gpu_ms (median)",
        title="Experiment 04 Sequential Indexing: Dispatch Time",
        output_path=CHARTS_DIR / "sequential_indexing_gpu_ms_vs_size.png",
    )

    print(f"Processed runs: {len(run_index)}")
    print(f"Latest run: {latest_run_id}")
    print(f"Wrote tables to: {TABLES_DIR}")
    print(f"Wrote charts to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
