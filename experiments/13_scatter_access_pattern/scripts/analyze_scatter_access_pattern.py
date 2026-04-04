#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "13_scatter_access_pattern"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"

VARIANT_ORDER = {
    "unique_permutation": 0,
    "random_collision_x4": 1,
    "clustered_hotset_32": 2,
}


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

    frame = frame[frame["experiment_id"] == EXPERIMENT_ID].copy()
    if frame.empty:
        raise ValueError("Input benchmark JSON has rows but no Experiment 13 entries.")

    numeric_columns = ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "dispatch_count" in frame.columns:
        frame["dispatch_count"] = pd.to_numeric(frame["dispatch_count"], errors="coerce")
    else:
        frame["dispatch_count"] = 1

    frame["variant"] = frame["variant"].fillna("unknown_variant").astype(str)
    if "notes" in frame.columns:
        frame["notes"] = frame["notes"].fillna("").astype(str)
    else:
        frame["notes"] = ""

    if "correctness_pass" in frame.columns:
        frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
    else:
        frame["correctness_pass"] = False

    note_maps = frame["notes"].apply(_parse_notes_field)
    for column_name, note_key in [
        ("distribution", "distribution"),
        ("seed", "seed"),
        ("collision_factor", "collision_factor"),
        ("hot_window_size", "hot_window_size"),
        ("hot_target_count", "hot_target_count"),
        ("active_target_count", "active_target_count"),
        ("max_expected_counter", "max_expected_counter"),
        ("target_capacity", "target_capacity"),
        ("logical_elements", "logical_elements"),
        ("physical_elements", "physical_elements"),
        ("physical_span_bytes", "physical_span_bytes"),
        ("allocated_span_bytes", "allocated_span_bytes"),
        ("bytes_per_logical_element", "bytes_per_logical_element"),
    ]:
        frame[column_name] = pd.to_numeric(note_maps.apply(lambda mapping: mapping.get(note_key)), errors="coerce")

    frame["distribution"] = frame["distribution"].fillna(frame["variant"]).astype(str)

    frame["logical_elements"] = frame["logical_elements"].fillna(frame["problem_size"])
    frame["logical_elements"] = pd.to_numeric(frame["logical_elements"], errors="coerce")
    if frame["logical_elements"].isna().any():
        raise ValueError("Could not determine logical_elements for one or more rows.")
    frame["logical_elements"] = frame["logical_elements"].astype(int)

    frame["physical_elements"] = frame["physical_elements"].fillna(frame["logical_elements"])
    frame["physical_elements"] = pd.to_numeric(frame["physical_elements"], errors="coerce")
    if frame["physical_elements"].isna().any():
        raise ValueError("Could not determine physical_elements for one or more rows.")
    frame["physical_elements"] = frame["physical_elements"].astype(int)

    frame["physical_span_bytes"] = frame["physical_span_bytes"].fillna(frame["logical_elements"] * 4)
    frame["physical_span_bytes"] = pd.to_numeric(frame["physical_span_bytes"], errors="coerce")
    if frame["physical_span_bytes"].isna().any():
        raise ValueError("Could not determine physical_span_bytes for one or more rows.")
    frame["physical_span_bytes"] = frame["physical_span_bytes"].astype(int)

    frame["allocated_span_bytes"] = frame["allocated_span_bytes"].fillna(frame["physical_span_bytes"])
    frame["allocated_span_bytes"] = pd.to_numeric(frame["allocated_span_bytes"], errors="coerce")
    if frame["allocated_span_bytes"].isna().any():
        raise ValueError("Could not determine allocated_span_bytes for one or more rows.")
    frame["allocated_span_bytes"] = frame["allocated_span_bytes"].astype(int)

    frame["bytes_per_logical_element"] = frame["bytes_per_logical_element"].fillna(12)
    frame["bytes_per_logical_element"] = pd.to_numeric(frame["bytes_per_logical_element"], errors="coerce")
    if frame["bytes_per_logical_element"].isna().any():
        raise ValueError("Could not determine bytes_per_logical_element for one or more rows.")
    frame["bytes_per_logical_element"] = frame["bytes_per_logical_element"].astype(int)

    for column_name, default_value in [
        ("seed", 0),
        ("collision_factor", 1),
        ("hot_window_size", 0),
        ("hot_target_count", 0),
        ("active_target_count", 0),
        ("max_expected_counter", 0),
        ("target_capacity", 0),
    ]:
        frame[column_name] = frame[column_name].fillna(default_value)
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
        if frame[column_name].isna().any():
            raise ValueError(f"Could not determine {column_name} for one or more rows.")
        frame[column_name] = frame[column_name].astype(int)

    frame["variant_order"] = frame["variant"].map(lambda variant: VARIANT_ORDER.get(variant, len(VARIANT_ORDER)))

    metadata = payload.get("metadata", {})
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
            "No benchmark_results.json or archived run JSON found for Experiment 13. "
            "Run data collection first."
        )

    frame, metadata = _load_frame(latest_run)
    return frame, metadata, latest_run


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["variant", "logical_elements"], as_index=False)
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
            end_to_end_ms_median=("end_to_end_ms", "median"),
            throughput_median=("throughput", "median"),
            distribution=("distribution", "first"),
            seed=("seed", "first"),
            collision_factor=("collision_factor", "first"),
            hot_window_size=("hot_window_size", "first"),
            hot_target_count=("hot_target_count", "first"),
            active_target_count=("active_target_count", "first"),
            max_expected_counter=("max_expected_counter", "first"),
            target_capacity=("target_capacity", "first"),
            physical_span_bytes=("physical_span_bytes", "first"),
            allocated_span_bytes=("allocated_span_bytes", "first"),
            bytes_per_logical_element=("bytes_per_logical_element", "first"),
            variant_order=("variant_order", "first"),
        )
        .sort_values(["variant_order", "logical_elements", "variant"])
        .reset_index(drop=True)
    )

    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    summary["p95_to_median_gpu_ms"] = summary["gpu_ms_p95"] / summary["gpu_ms_median"]
    summary["p95_to_median_gbps"] = summary["gbps_p95"] / summary["gbps_median"]
    summary["active_target_ratio"] = summary["active_target_count"] / summary["target_capacity"]
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = summary[summary["variant"] == "unique_permutation"]
    if baseline_rows.empty:
        raise ValueError("Could not find unique_permutation baseline in summary data.")

    baseline = baseline_rows.iloc[0]
    relative = summary.copy()
    relative["slowdown_vs_unique_permutation"] = relative["gpu_ms_median"] / float(baseline["gpu_ms_median"])
    relative["delta_gpu_ms_vs_unique_permutation_pct"] = (
        (relative["gpu_ms_median"] - float(baseline["gpu_ms_median"])) / float(baseline["gpu_ms_median"])
    ) * 100.0
    relative["gbps_ratio_vs_unique_permutation"] = relative["gbps_median"] / float(baseline["gbps_median"])
    relative["proxy_bandwidth_loss_pct"] = (1.0 - relative["gbps_ratio_vs_unique_permutation"]) * 100.0

    columns = [
        "variant",
        "logical_elements",
        "gpu_ms_median",
        "slowdown_vs_unique_permutation",
        "delta_gpu_ms_vs_unique_permutation_pct",
        "gbps_median",
        "gbps_ratio_vs_unique_permutation",
        "proxy_bandwidth_loss_pct",
    ]
    return relative[columns].sort_values(["logical_elements", "variant"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "variant",
            "logical_elements",
            "gpu_ms_p95",
            "p95_to_median_gpu_ms",
            "gpu_ms_cv",
            "gbps_p95",
            "p95_to_median_gbps",
            "gbps_cv",
            "sample_count",
            "correctness_pass_rate",
        ]
    ].copy()
    return stability.sort_values(["logical_elements", "variant"]).reset_index(drop=True)


def _build_contention(summary: pd.DataFrame) -> pd.DataFrame:
    contention = summary[
        [
            "variant",
            "logical_elements",
            "collision_factor",
            "hot_window_size",
            "hot_target_count",
            "active_target_count",
            "active_target_ratio",
            "max_expected_counter",
            "target_capacity",
            "physical_span_bytes",
        ]
    ].copy()
    return contention.sort_values(["logical_elements", "variant"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 13 scatter access pattern data.")
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
    contention = _build_contention(summary)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _write_table(summary, TABLES_DIR / "scatter_access_pattern_summary.csv")
    _write_table(relative, TABLES_DIR / "scatter_access_pattern_relative.csv")
    _write_table(stability, TABLES_DIR / "scatter_access_pattern_stability.csv")
    _write_table(contention, TABLES_DIR / "scatter_access_pattern_contention.csv")

    rows = int(frame.shape[0])
    variants = int(summary.shape[0])
    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(f"[ok] Wrote Experiment 13 analysis tables from {source_label} ({rows} rows, {variants} variants on {gpu_name}).")


if __name__ == "__main__":
    main()
