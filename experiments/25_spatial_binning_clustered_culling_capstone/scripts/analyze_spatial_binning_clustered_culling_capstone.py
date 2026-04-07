#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "25_spatial_binning_clustered_culling_capstone"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"

STRATEGY_ORDER = {
    "global_append": 0,
    "coherent_append": 1,
}

DISTRIBUTION_ORDER = {
    "uniform_sparse": 0,
    "uniform_dense": 1,
    "clustered": 2,
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
        raise ValueError("Input benchmark JSON has rows but no Experiment 25 entries.")

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
        ("strategy", "strategy"),
        ("distribution", "distribution"),
        ("seed", "seed"),
        ("bin_count", "bin_count"),
        ("active_bin_count", "active_bin_count"),
        ("max_bin_load", "max_bin_load"),
        ("mean_active_bin_load", "mean_active_bin_load"),
        ("host_sorted", "host_sorted"),
        ("local_size_x", "local_size_x"),
        ("group_count_x", "group_count_x"),
        ("entity_count", "entity_count"),
        ("input_span_bytes", "input_span_bytes"),
        ("counter_span_bytes", "counter_span_bytes"),
        ("bin_list_span_bytes", "bin_list_span_bytes"),
        ("scratch_size_bytes", "scratch_size_bytes"),
        ("estimated_global_total_bytes", "estimated_global_total_bytes"),
    ]:
        frame[column_name] = note_maps.apply(lambda mapping: mapping.get(note_key))

    frame["strategy"] = frame["strategy"].fillna(frame["variant"].str.extract(r"^(global_append|coherent_append)")[0])
    frame["distribution"] = frame["distribution"].fillna(
        frame["variant"].str.extract(r"_(uniform_sparse|uniform_dense|clustered)$")[0]
    )

    for column_name, default_value in [
        ("seed", 0),
        ("bin_count", 64),
        ("active_bin_count", 0),
        ("max_bin_load", 0),
        ("mean_active_bin_load", 0.0),
        ("host_sorted", 0),
        ("local_size_x", 256),
        ("group_count_x", 0),
        ("entity_count", 0),
        ("input_span_bytes", 0),
        ("counter_span_bytes", 0),
        ("bin_list_span_bytes", 0),
        ("scratch_size_bytes", 0),
        ("estimated_global_total_bytes", 0),
    ]:
        frame[column_name] = pd.to_numeric(frame[column_name].fillna(default_value), errors="coerce")
        if frame[column_name].isna().any():
            raise ValueError(f"Could not determine {column_name} for one or more rows.")

    if frame["strategy"].isna().any() or frame["distribution"].isna().any():
        raise ValueError("Could not determine strategy or distribution for one or more rows.")

    frame["strategy_order"] = frame["strategy"].map(lambda value: STRATEGY_ORDER.get(str(value), len(STRATEGY_ORDER)))
    frame["distribution_order"] = frame["distribution"].map(
        lambda value: DISTRIBUTION_ORDER.get(str(value), len(DISTRIBUTION_ORDER))
    )

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
            "No benchmark_results.json or archived run JSON found for Experiment 25. "
            "Run data collection first."
        )

    frame, metadata = _load_frame(latest_run)
    return frame, metadata, latest_run


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["strategy", "distribution", "entity_count"], as_index=False)
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
            bin_count=("bin_count", "first"),
            active_bin_count=("active_bin_count", "first"),
            max_bin_load=("max_bin_load", "first"),
            mean_active_bin_load=("mean_active_bin_load", "first"),
            host_sorted=("host_sorted", "first"),
            local_size_x=("local_size_x", "first"),
            group_count_x=("group_count_x", "first"),
            input_span_bytes=("input_span_bytes", "first"),
            counter_span_bytes=("counter_span_bytes", "first"),
            bin_list_span_bytes=("bin_list_span_bytes", "first"),
            scratch_size_bytes=("scratch_size_bytes", "first"),
            estimated_global_total_bytes=("estimated_global_total_bytes", "first"),
            strategy_order=("strategy_order", "first"),
            distribution_order=("distribution_order", "first"),
        )
        .sort_values(["distribution_order", "strategy_order"])
        .reset_index(drop=True)
    )

    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    summary["p95_to_median_gpu_ms"] = summary["gpu_ms_p95"] / summary["gpu_ms_median"]
    summary["p95_to_median_gbps"] = summary["gbps_p95"] / summary["gbps_median"]
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    global_rows = (
        summary[summary["strategy"] == "global_append"][
            ["distribution", "entity_count", "gpu_ms_median", "gbps_median", "throughput_median"]
        ]
        .rename(
            columns={
                "gpu_ms_median": "global_gpu_ms_median",
                "gbps_median": "global_gbps_median",
                "throughput_median": "global_throughput_median",
            }
        )
        .reset_index(drop=True)
    )
    coherent_rows = (
        summary[summary["strategy"] == "coherent_append"][
            ["distribution", "entity_count", "gpu_ms_median", "gbps_median", "throughput_median"]
        ]
        .rename(
            columns={
                "gpu_ms_median": "coherent_gpu_ms_median",
                "gbps_median": "coherent_gbps_median",
                "throughput_median": "coherent_throughput_median",
            }
        )
        .reset_index(drop=True)
    )

    relative = global_rows.merge(coherent_rows, on=["distribution", "entity_count"], how="inner")
    if relative.empty:
        raise ValueError("Could not build global-versus-coherent relative table.")

    relative["coherent_speedup_vs_global"] = relative["global_gpu_ms_median"] / relative["coherent_gpu_ms_median"]
    relative["coherent_gpu_delta_pct_vs_global"] = (
        (relative["coherent_gpu_ms_median"] - relative["global_gpu_ms_median"]) / relative["global_gpu_ms_median"]
    ) * 100.0
    relative["coherent_gbps_ratio_vs_global"] = relative["coherent_gbps_median"] / relative["global_gbps_median"]
    relative["coherent_throughput_ratio_vs_global"] = (
        relative["coherent_throughput_median"] / relative["global_throughput_median"]
    )
    relative["distribution_order"] = relative["distribution"].map(
        lambda value: DISTRIBUTION_ORDER.get(str(value), len(DISTRIBUTION_ORDER))
    )
    return relative.sort_values(["distribution_order"]).drop(columns=["distribution_order"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "strategy",
            "distribution",
            "entity_count",
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
    return stability.sort_values(["distribution", "strategy"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 25 spatial binning capstone data.")
    parser.add_argument("--skip-current", action="store_true")
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current)

    summary = _build_summary(frame)
    relative = _build_relative(summary)
    stability = _build_stability(summary)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _write_table(summary, TABLES_DIR / "spatial_binning_clustered_culling_capstone_summary.csv")
    _write_table(relative, TABLES_DIR / "spatial_binning_clustered_culling_capstone_relative.csv")
    _write_table(stability, TABLES_DIR / "spatial_binning_clustered_culling_capstone_stability.csv")

    rows = int(frame.shape[0])
    variants = int(summary.shape[0])
    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(f"[ok] Wrote Experiment 25 analysis tables from {source_label} ({rows} rows, {variants} cases on {gpu_name}).")


if __name__ == "__main__":
    main()
