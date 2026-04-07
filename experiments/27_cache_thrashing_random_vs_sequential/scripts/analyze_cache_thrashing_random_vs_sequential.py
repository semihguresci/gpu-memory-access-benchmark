#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "27_cache_thrashing_random_vs_sequential"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"

VARIANT_ORDER = {
    "sequential": 0,
    "block_shuffled": 1,
    "random": 2,
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
        raise ValueError("Input benchmark JSON has rows but no Experiment 27 entries.")

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
    def _first_note_value(mapping: dict[str, str], *keys: str) -> str | None:
        for key in keys:
            value = mapping.get(key)
            if value is not None and value != "":
                return value
        return None

    frame["access_pattern"] = note_maps.apply(lambda mapping: _first_note_value(mapping, "access_pattern", "pattern"))
    frame["pattern_seed"] = note_maps.apply(lambda mapping: _first_note_value(mapping, "pattern_seed", "seed"))
    frame["seed"] = note_maps.apply(lambda mapping: _first_note_value(mapping, "seed", "pattern_seed"))
    frame["block_size"] = note_maps.apply(
        lambda mapping: _first_note_value(mapping, "block_size", "block_shuffle_width")
    )
    frame["local_size_x"] = note_maps.apply(lambda mapping: _first_note_value(mapping, "local_size_x"))
    frame["group_count_x"] = note_maps.apply(lambda mapping: _first_note_value(mapping, "group_count_x"))
    frame["logical_elements"] = note_maps.apply(lambda mapping: _first_note_value(mapping, "logical_elements"))
    frame["working_set_bytes"] = note_maps.apply(lambda mapping: _first_note_value(mapping, "working_set_bytes"))
    frame["source_span_bytes"] = note_maps.apply(lambda mapping: _first_note_value(mapping, "source_span_bytes"))
    frame["permutation_span_bytes"] = note_maps.apply(
        lambda mapping: _first_note_value(mapping, "permutation_span_bytes")
    )
    frame["destination_span_bytes"] = note_maps.apply(
        lambda mapping: _first_note_value(mapping, "destination_span_bytes")
    )
    frame["total_allocation_bytes"] = note_maps.apply(
        lambda mapping: _first_note_value(mapping, "total_allocation_bytes")
    )
    frame["bytes_per_logical_element"] = note_maps.apply(
        lambda mapping: _first_note_value(mapping, "bytes_per_logical_element", "payload_bytes_per_element")
    )

    frame["access_pattern"] = frame["access_pattern"].fillna(frame["variant"]).astype(str)
    frame["logical_elements"] = frame["logical_elements"].fillna(frame["problem_size"])
    frame["logical_elements"] = pd.to_numeric(frame["logical_elements"], errors="coerce")
    if frame["logical_elements"].isna().any():
        raise ValueError("Could not determine logical_elements for one or more rows.")
    frame["logical_elements"] = frame["logical_elements"].astype(int)

    for column_name, default_value in [
        ("pattern_seed", 0),
        ("seed", 0),
        ("block_size", 256),
        ("local_size_x", 256),
        ("group_count_x", 0),
        ("working_set_bytes", 0),
        ("source_span_bytes", 0),
        ("permutation_span_bytes", 0),
        ("destination_span_bytes", 0),
        ("total_allocation_bytes", 0),
        ("bytes_per_logical_element", 12),
    ]:
        frame[column_name] = pd.to_numeric(frame[column_name].fillna(default_value), errors="coerce")
        if frame[column_name].isna().any():
            raise ValueError(f"Could not determine {column_name} for one or more rows.")
        frame[column_name] = frame[column_name].astype(int)

    frame["working_set_bytes"] = frame["working_set_bytes"].where(frame["working_set_bytes"] > 0, frame["source_span_bytes"])
    frame["source_span_bytes"] = frame["source_span_bytes"].where(frame["source_span_bytes"] > 0, frame["working_set_bytes"])
    frame["permutation_span_bytes"] = frame["permutation_span_bytes"].where(
        frame["permutation_span_bytes"] > 0, frame["working_set_bytes"]
    )
    frame["destination_span_bytes"] = frame["destination_span_bytes"].where(
        frame["destination_span_bytes"] > 0, frame["working_set_bytes"]
    )
    frame["total_allocation_bytes"] = frame["total_allocation_bytes"].where(
        frame["total_allocation_bytes"] > 0, frame["working_set_bytes"] * 3
    )

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
            "No benchmark_results.json or archived run JSON found for Experiment 27. "
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
            access_pattern=("access_pattern", "first"),
            pattern_seed=("pattern_seed", "first"),
            seed=("seed", "first"),
            block_size=("block_size", "first"),
            local_size_x=("local_size_x", "first"),
            group_count_x=("group_count_x", "first"),
            working_set_bytes=("working_set_bytes", "first"),
            source_span_bytes=("source_span_bytes", "first"),
            permutation_span_bytes=("permutation_span_bytes", "first"),
            destination_span_bytes=("destination_span_bytes", "first"),
            total_allocation_bytes=("total_allocation_bytes", "first"),
            bytes_per_logical_element=("bytes_per_logical_element", "first"),
            variant_order=("variant_order", "first"),
        )
        .sort_values(["variant_order", "logical_elements", "variant"])
        .reset_index(drop=True)
    )

    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = summary[summary["variant"] == "sequential"]
    if baseline_rows.empty:
        raise ValueError("Could not find sequential baseline in summary data.")

    baseline = baseline_rows[["logical_elements", "gpu_ms_median", "gbps_median", "throughput_median"]].rename(
        columns={
            "gpu_ms_median": "baseline_gpu_ms_median",
            "gbps_median": "baseline_gbps_median",
            "throughput_median": "baseline_throughput_median",
        }
    )

    relative = summary.merge(baseline, on="logical_elements", how="inner")
    if relative.empty:
        raise ValueError("Could not build relative table for Experiment 27.")

    relative["slowdown_vs_sequential"] = relative["gpu_ms_median"] / relative["baseline_gpu_ms_median"]
    relative["speedup_vs_sequential"] = relative["baseline_gpu_ms_median"] / relative["gpu_ms_median"]
    relative["delta_gpu_ms_vs_sequential_pct"] = (
        (relative["gpu_ms_median"] - relative["baseline_gpu_ms_median"]) / relative["baseline_gpu_ms_median"]
    ) * 100.0
    relative["gbps_ratio_vs_sequential"] = relative["gbps_median"] / relative["baseline_gbps_median"]
    relative["throughput_ratio_vs_sequential"] = (
        relative["throughput_median"] / relative["baseline_throughput_median"]
    )

    columns = [
        "variant",
        "logical_elements",
        "gpu_ms_median",
        "slowdown_vs_sequential",
        "speedup_vs_sequential",
        "delta_gpu_ms_vs_sequential_pct",
        "gbps_median",
        "gbps_ratio_vs_sequential",
        "throughput_median",
        "throughput_ratio_vs_sequential",
    ]
    return relative[columns].sort_values(["logical_elements", "variant"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "variant",
            "logical_elements",
            "gpu_ms_p95",
            "gpu_ms_median",
            "gpu_ms_cv",
            "gbps_p95",
            "gbps_median",
            "gbps_cv",
            "sample_count",
            "correctness_pass_rate",
        ]
    ].copy()
    stability["p95_to_median_gpu_ms"] = stability["gpu_ms_p95"] / stability["gpu_ms_median"]
    stability["p95_to_median_gbps"] = stability["gbps_p95"] / stability["gbps_median"]
    stability = stability[
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
    ]
    return stability.sort_values(["logical_elements", "variant"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 27 cache thrashing data.")
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
    _write_table(summary, TABLES_DIR / "cache_thrashing_random_vs_sequential_summary.csv")
    _write_table(relative, TABLES_DIR / "cache_thrashing_random_vs_sequential_relative.csv")
    _write_table(stability, TABLES_DIR / "cache_thrashing_random_vs_sequential_stability.csv")

    rows = int(frame.shape[0])
    variants = int(summary.shape[0])
    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(
        f"[ok] Wrote Experiment 27 analysis tables from {source_label} "
        f"({rows} rows, {variants} variants on {gpu_name})."
    )


if __name__ == "__main__":
    main()
