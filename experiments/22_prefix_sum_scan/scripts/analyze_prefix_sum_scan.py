#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "22_prefix_sum_scan"
BYTES_PER_ELEMENT = 4.0

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"


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


def _latest_run_path(runs_dir: Path) -> Path | None:
    candidates = sorted(path for path in runs_dir.rglob("*.json") if path.is_file()) if runs_dir.exists() else []
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _load_payload(input_path: Path) -> dict:
    return json.loads(input_path.read_text(encoding="utf-8"))


def _load_frame(input_path: Path) -> tuple[pd.DataFrame, dict]:
    payload = _load_payload(input_path)
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input benchmark JSON has no rows[] payload.")

    frame = pd.DataFrame(rows)
    if "experiment_id" not in frame.columns:
        raise ValueError("Input benchmark JSON is missing experiment_id rows.")
    frame = frame[frame["experiment_id"].astype(str) == EXPERIMENT_ID].copy()
    if frame.empty:
        raise ValueError("Input benchmark JSON has rows but no Experiment 22 entries.")

    for column in ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps", "dispatch_count"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["problem_size", "gpu_ms", "gbps"])
    frame["problem_size"] = frame["problem_size"].astype(int)
    frame["iteration"] = frame["iteration"].fillna(0).astype(int)
    frame["dispatch_count"] = frame["dispatch_count"].fillna(1).astype(int)
    frame["variant"] = frame["variant"].fillna("unknown_variant").astype(str)
    if "correctness_pass" in frame.columns:
        frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
    else:
        frame["correctness_pass"] = False
    if "notes" in frame.columns:
        frame["notes"] = frame["notes"].fillna("").astype(str)
    else:
        frame["notes"] = ""
    frame["size_bytes"] = frame["problem_size"] * BYTES_PER_ELEMENT
    frame["size_mib"] = frame["size_bytes"] / (1024.0 * 1024.0)

    metadata = payload.get("metadata", {})
    frame["exported_at_utc"] = str(metadata.get("exported_at_utc", ""))
    frame["gpu_name"] = str(metadata.get("gpu_name", "unknown_gpu"))
    frame["driver_version"] = str(metadata.get("driver_version", "unknown_driver"))
    frame["vulkan_api_version"] = str(metadata.get("vulkan_api_version", "unknown_api"))
    frame["validation_enabled"] = bool(metadata.get("validation_enabled", False))
    frame["device_id"] = (
        f"{frame['gpu_name'].iloc[0]} | drv {frame['driver_version'].iloc[0]} | vk {frame['vulkan_api_version'].iloc[0]}"
    )
    return frame, metadata


def _load_source_frame(skip_current: bool) -> tuple[pd.DataFrame, dict, Path]:
    if not skip_current and CURRENT_JSON.exists():
        frame, metadata = _load_frame(CURRENT_JSON)
        return frame, metadata, CURRENT_JSON

    latest_run = _latest_run_path(RUNS_DIR)
    if latest_run is None:
        if CURRENT_JSON.exists():
            frame, metadata = _load_frame(CURRENT_JSON)
            return frame, metadata, CURRENT_JSON
        raise FileNotFoundError(
            "No benchmark_results.json or archived run JSON found for Experiment 22. Run data collection first."
        )

    frame, metadata = _load_frame(latest_run)
    return frame, metadata, latest_run


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["variant", "problem_size", "size_bytes", "size_mib"], as_index=False)
        .agg(
            sample_count=("gpu_ms", "count"),
            correctness_pass_rate=("correctness_pass", "mean"),
            gpu_ms_mean=("gpu_ms", "mean"),
            gpu_ms_median=("gpu_ms", "median"),
            gpu_ms_p95=("gpu_ms", _quantile_95),
            gpu_ms_std=("gpu_ms", lambda series: float(series.std(ddof=0))),
            gbps_mean=("gbps", "mean"),
            gbps_median=("gbps", "median"),
            gbps_p95=("gbps", _quantile_95),
            gbps_std=("gbps", lambda series: float(series.std(ddof=0))),
            throughput_median=("throughput", "median"),
            end_to_end_ms_median=("end_to_end_ms", "median"),
        )
        .sort_values(["size_bytes", "variant"])
        .reset_index(drop=True)
    )
    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    summary["p95_to_median_gpu_ms"] = summary["gpu_ms_p95"] / summary["gpu_ms_median"]
    summary["p95_to_median_gbps"] = summary["gbps_p95"] / summary["gbps_median"]
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        summary[summary["variant"] == "items_per_thread_1"][
            ["problem_size", "size_bytes", "size_mib", "gpu_ms_median", "gbps_median", "throughput_median"]
        ]
        .rename(
            columns={
                "gpu_ms_median": "baseline_gpu_ms_median",
                "gbps_median": "baseline_gbps_median",
                "throughput_median": "baseline_throughput_median",
            }
        )
        .reset_index(drop=True)
    )
    relative = summary.merge(baseline, on=["problem_size", "size_bytes", "size_mib"], how="inner")
    relative["speedup_vs_baseline"] = relative["baseline_gpu_ms_median"] / relative["gpu_ms_median"]
    relative["gpu_delta_pct_vs_baseline"] = (
        (relative["gpu_ms_median"] - relative["baseline_gpu_ms_median"]) / relative["baseline_gpu_ms_median"]
    ) * 100.0
    relative["gbps_ratio_vs_baseline"] = relative["gbps_median"] / relative["baseline_gbps_median"]
    relative["throughput_ratio_vs_baseline"] = (
        relative["throughput_median"] / relative["baseline_throughput_median"]
    )
    return relative.sort_values(["size_bytes", "variant"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        [
            "variant",
            "problem_size",
            "gpu_ms_p95",
            "p95_to_median_gpu_ms",
            "gpu_ms_cv",
            "gbps_p95",
            "p95_to_median_gbps",
            "gbps_cv",
            "sample_count",
            "correctness_pass_rate",
        ]
    ].sort_values(["problem_size", "variant"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 22 prefix sum scan data.")
    parser.add_argument(
        "--skip-current",
        action="store_true",
        help="Prefer the latest archived run under runs/ instead of results/tables/benchmark_results.json.",
    )
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current)
    summary = _build_summary(frame)
    relative = _build_relative(summary)
    stability = _build_stability(summary)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _write_table(summary, TABLES_DIR / "prefix_sum_scan_summary.csv")
    _write_table(relative, TABLES_DIR / "prefix_sum_scan_relative.csv")
    _write_table(stability, TABLES_DIR / "prefix_sum_scan_stability.csv")

    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(f"[ok] Wrote Experiment 22 analysis tables from {source_label} ({frame.shape[0]} rows on {gpu_name}).")


if __name__ == "__main__":
    main()
