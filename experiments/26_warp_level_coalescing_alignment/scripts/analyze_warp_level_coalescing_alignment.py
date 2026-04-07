#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "26_warp_level_coalescing_alignment"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"

ALIGNMENT_ORDER = {
    0: 0,
    4: 1,
    8: 2,
    16: 3,
    32: 4,
    64: 5,
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


def _alignment_bytes_from_variant(variant: str) -> int | None:
    if variant == "aligned":
        return 0

    match = re.search(r"offset_(\d+)b$", variant)
    if match is None:
        return None
    return int(match.group(1))


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
        raise ValueError("Input benchmark JSON has rows but no Experiment 26 entries.")

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
        ("alignment_offset_bytes", "alignment_offset_bytes"),
        ("alignment_offset_elements", "alignment_offset_elements"),
        ("max_alignment_offset_bytes", "max_alignment_offset_bytes"),
        ("max_alignment_offset_elements", "max_alignment_offset_elements"),
        ("local_size_x", "local_size_x"),
        ("group_count_x", "group_count_x"),
        ("logical_elements", "logical_elements"),
        ("physical_input_elements", "physical_input_elements"),
        ("physical_input_span_bytes", "physical_input_span_bytes"),
        ("output_span_bytes", "output_span_bytes"),
        ("logical_bytes_touched", "logical_bytes_touched"),
        ("bytes_per_logical_element", "bytes_per_logical_element"),
        ("alignment_baseline", "alignment_baseline"),
        ("access_pattern", "access_pattern"),
        ("validation_mode", "validation_mode"),
    ]:
        frame[column_name] = note_maps.apply(lambda mapping: mapping.get(note_key))

    frame["alignment_offset_bytes"] = frame["alignment_offset_bytes"].fillna(
        frame["variant"].map(_alignment_bytes_from_variant)
    )

    frame["alignment_offset_bytes"] = pd.to_numeric(frame["alignment_offset_bytes"], errors="coerce")
    frame["alignment_offset_elements"] = pd.to_numeric(frame["alignment_offset_elements"], errors="coerce")
    frame["max_alignment_offset_bytes"] = pd.to_numeric(frame["max_alignment_offset_bytes"], errors="coerce")
    frame["max_alignment_offset_elements"] = pd.to_numeric(frame["max_alignment_offset_elements"], errors="coerce")
    frame["local_size_x"] = pd.to_numeric(frame["local_size_x"], errors="coerce")
    frame["group_count_x"] = pd.to_numeric(frame["group_count_x"], errors="coerce")
    frame["logical_elements"] = pd.to_numeric(frame["logical_elements"], errors="coerce")
    frame["physical_input_elements"] = pd.to_numeric(frame["physical_input_elements"], errors="coerce")
    frame["physical_input_span_bytes"] = pd.to_numeric(frame["physical_input_span_bytes"], errors="coerce")
    frame["output_span_bytes"] = pd.to_numeric(frame["output_span_bytes"], errors="coerce")
    frame["logical_bytes_touched"] = pd.to_numeric(frame["logical_bytes_touched"], errors="coerce")
    frame["bytes_per_logical_element"] = pd.to_numeric(frame["bytes_per_logical_element"], errors="coerce")

    frame["alignment_baseline"] = frame["alignment_baseline"].fillna("").astype(str).str.lower().isin(["1", "true", "yes"])
    frame["access_pattern"] = frame["access_pattern"].fillna("contiguous_shifted").astype(str)
    frame["validation_mode"] = frame["validation_mode"].fillna("transform_value_u32").astype(str)

    frame["problem_size"] = frame["problem_size"].fillna(0).astype(int)
    frame["iteration"] = frame["iteration"].fillna(0).astype(int)
    frame["dispatch_count"] = frame["dispatch_count"].fillna(1).astype(int)
    frame["alignment_offset_bytes"] = frame["alignment_offset_bytes"].fillna(0).astype(int)
    frame["alignment_offset_elements"] = frame["alignment_offset_elements"].fillna(frame["alignment_offset_bytes"] // 4).astype(int)
    frame["max_alignment_offset_bytes"] = frame["max_alignment_offset_bytes"].fillna(64).astype(int)
    frame["max_alignment_offset_elements"] = frame["max_alignment_offset_elements"].fillna(16).astype(int)
    frame["local_size_x"] = frame["local_size_x"].fillna(256).astype(int)
    frame["group_count_x"] = frame["group_count_x"].fillna(0).astype(int)
    frame["logical_elements"] = frame["logical_elements"].fillna(frame["problem_size"]).astype(int)
    frame["physical_input_elements"] = frame["physical_input_elements"].fillna(
        frame["logical_elements"] + frame["max_alignment_offset_elements"]
    ).astype(int)
    frame["physical_input_span_bytes"] = frame["physical_input_span_bytes"].fillna(
        frame["physical_input_elements"] * 4
    ).astype(int)
    frame["output_span_bytes"] = frame["output_span_bytes"].fillna(frame["logical_elements"] * 4).astype(int)
    frame["logical_bytes_touched"] = frame["logical_bytes_touched"].fillna(frame["logical_elements"] * 8).astype(int)
    frame["bytes_per_logical_element"] = frame["bytes_per_logical_element"].fillna(8).astype(int)

    frame["alignment_order"] = frame["alignment_offset_bytes"].map(
        lambda value: ALIGNMENT_ORDER.get(int(value), len(ALIGNMENT_ORDER))
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
            "No benchmark_results.json or archived run JSON found for Experiment 26. Run data collection first."
        )

    frame, metadata = _load_frame(latest_run)
    return frame, metadata, latest_run


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["alignment_offset_bytes", "variant", "problem_size"], as_index=False)
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
            alignment_offset_elements=("alignment_offset_elements", "first"),
            max_alignment_offset_bytes=("max_alignment_offset_bytes", "first"),
            max_alignment_offset_elements=("max_alignment_offset_elements", "first"),
            local_size_x=("local_size_x", "first"),
            group_count_x=("group_count_x", "first"),
            logical_elements=("logical_elements", "first"),
            physical_input_elements=("physical_input_elements", "first"),
            physical_input_span_bytes=("physical_input_span_bytes", "first"),
            output_span_bytes=("output_span_bytes", "first"),
            logical_bytes_touched=("logical_bytes_touched", "first"),
            bytes_per_logical_element=("bytes_per_logical_element", "first"),
            alignment_baseline=("alignment_baseline", "first"),
            access_pattern=("access_pattern", "first"),
            validation_mode=("validation_mode", "first"),
            alignment_order=("alignment_order", "first"),
        )
        .sort_values(["alignment_order", "variant", "problem_size"])
        .reset_index(drop=True)
    )

    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    summary["p95_to_median_gpu_ms"] = summary["gpu_ms_p95"] / summary["gpu_ms_median"]
    summary["p95_to_median_gbps"] = summary["gbps_p95"] / summary["gbps_median"]
    summary["useful_bytes"] = summary["logical_elements"] * summary["bytes_per_logical_element"]
    summary["input_padding_bytes"] = summary["physical_input_span_bytes"] - (summary["logical_elements"] * 4)
    summary["working_set_bytes"] = summary["physical_input_span_bytes"] + summary["output_span_bytes"]
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        summary[summary["alignment_baseline"]][
            ["problem_size", "gpu_ms_median", "gbps_median", "throughput_median"]
        ]
        .rename(
            columns={
                "gpu_ms_median": "aligned_gpu_ms_median",
                "gbps_median": "aligned_gbps_median",
                "throughput_median": "aligned_throughput_median",
            }
        )
        .reset_index(drop=True)
    )
    relative = summary.merge(baseline, on="problem_size", how="left")
    if relative["aligned_gpu_ms_median"].isna().any():
        raise ValueError("Could not build relative table against aligned baseline.")

    relative["slowdown_vs_aligned"] = relative["gpu_ms_median"] / relative["aligned_gpu_ms_median"]
    relative["delta_gpu_ms_vs_aligned_pct"] = (
        (relative["gpu_ms_median"] - relative["aligned_gpu_ms_median"]) / relative["aligned_gpu_ms_median"]
    ) * 100.0
    relative["gbps_ratio_vs_aligned"] = relative["gbps_median"] / relative["aligned_gbps_median"]
    relative["gbps_delta_pct_vs_aligned"] = (1.0 - relative["gbps_ratio_vs_aligned"]) * 100.0
    relative["throughput_ratio_vs_aligned"] = relative["throughput_median"] / relative["aligned_throughput_median"]

    columns = [
        "alignment_offset_bytes",
        "variant",
        "problem_size",
        "gpu_ms_median",
        "slowdown_vs_aligned",
        "delta_gpu_ms_vs_aligned_pct",
        "gbps_median",
        "gbps_ratio_vs_aligned",
        "gbps_delta_pct_vs_aligned",
        "throughput_median",
        "throughput_ratio_vs_aligned",
        "aligned_gpu_ms_median",
        "aligned_gbps_median",
        "aligned_throughput_median",
    ]
    return relative[columns].sort_values(["alignment_offset_bytes", "variant", "problem_size"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "alignment_offset_bytes",
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
    ].copy()
    return stability.sort_values(["alignment_offset_bytes", "variant", "problem_size"]).reset_index(drop=True)


def _build_footprint(summary: pd.DataFrame) -> pd.DataFrame:
    footprint = summary[
        [
            "alignment_offset_bytes",
            "variant",
            "problem_size",
            "alignment_offset_elements",
            "max_alignment_offset_bytes",
            "max_alignment_offset_elements",
            "local_size_x",
            "group_count_x",
            "logical_elements",
            "physical_input_elements",
            "physical_input_span_bytes",
            "output_span_bytes",
            "logical_bytes_touched",
            "bytes_per_logical_element",
            "input_padding_bytes",
            "working_set_bytes",
            "alignment_baseline",
            "access_pattern",
            "validation_mode",
        ]
    ].copy()
    return footprint.sort_values(["alignment_offset_bytes", "variant", "problem_size"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 26 warp-level coalescing alignment data.")
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
    footprint = _build_footprint(summary)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _write_table(summary, TABLES_DIR / "warp_level_coalescing_alignment_summary.csv")
    _write_table(relative, TABLES_DIR / "warp_level_coalescing_alignment_relative.csv")
    _write_table(stability, TABLES_DIR / "warp_level_coalescing_alignment_stability.csv")
    _write_table(footprint, TABLES_DIR / "warp_level_coalescing_alignment_footprint.csv")

    rows = int(frame.shape[0])
    variants = int(summary.shape[0])
    resolved_source = source_path.resolve()
    try:
        source_label = resolved_source.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        source_label = resolved_source.as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(f"[ok] Wrote Experiment 26 analysis tables from {source_label} ({rows} rows, {variants} cases on {gpu_name}).")


if __name__ == "__main__":
    main()
