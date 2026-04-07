#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "24_stream_compaction"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"

IMPLEMENTATION_ORDER = {
    "global_atomic_append": 0,
    "three_stage": 1,
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
        raise ValueError("Input benchmark JSON has rows but no Experiment 24 entries.")

    for column in ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps", "dispatch_count"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

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
    for column_name, note_key, default_value in [
        ("implementation", "implementation", ""),
        ("valid_ratio_percent", "valid_ratio_percent", 0),
        ("valid_count", "valid_count", 0),
        ("logical_elements", "logical_elements", 0),
        ("block_count", "block_count", 0),
        ("local_size_x", "local_size_x", 256),
        ("stable_ordering", "stable_ordering", "false"),
        ("pattern_seed", "pattern_seed", 0),
        ("scratch_size_bytes", "scratch_size_bytes", 0),
        ("estimated_total_bytes", "estimated_total_bytes", 0),
    ]:
        frame[column_name] = note_maps.apply(lambda mapping: mapping.get(note_key, default_value))

    frame["implementation"] = frame["implementation"].replace("", pd.NA).fillna(
        frame["variant"].str.extract(r"^(global_atomic_append|three_stage)")[0]
    )
    if frame["implementation"].isna().any():
        raise ValueError("Could not determine implementation for one or more rows.")

    for numeric_column in [
        "valid_ratio_percent",
        "valid_count",
        "logical_elements",
        "block_count",
        "local_size_x",
        "pattern_seed",
        "scratch_size_bytes",
        "estimated_total_bytes",
    ]:
        frame[numeric_column] = pd.to_numeric(frame[numeric_column], errors="coerce")
        if frame[numeric_column].isna().any():
            raise ValueError(f"Could not determine {numeric_column} for one or more rows.")
        frame[numeric_column] = frame[numeric_column].astype(int)

    frame["stable_ordering"] = frame["stable_ordering"].astype(str).str.lower().isin(["1", "true", "yes"])
    frame["implementation_order"] = frame["implementation"].map(
        lambda value: IMPLEMENTATION_ORDER.get(str(value), len(IMPLEMENTATION_ORDER))
    )

    return frame, payload.get("metadata", {})


def _load_source_frame(skip_current: bool) -> tuple[pd.DataFrame, dict, Path]:
    if not skip_current and CURRENT_JSON.exists():
        frame, metadata = _load_frame(CURRENT_JSON)
        return frame, metadata, CURRENT_JSON

    latest_run = _latest_run_path(RUNS_DIR)
    if latest_run is None:
        if CURRENT_JSON.exists():
            frame, metadata = _load_frame(CURRENT_JSON)
            return frame, metadata, CURRENT_JSON
        raise FileNotFoundError("No benchmark_results.json or archived run JSON found for Experiment 24.")

    frame, metadata = _load_frame(latest_run)
    return frame, metadata, latest_run


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["implementation", "valid_ratio_percent", "logical_elements"], as_index=False)
        .agg(
            sample_count_rows=("gpu_ms", "count"),
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
            valid_count=("valid_count", "first"),
            block_count=("block_count", "first"),
            local_size_x=("local_size_x", "first"),
            stable_ordering=("stable_ordering", "first"),
            pattern_seed=("pattern_seed", "first"),
            scratch_size_bytes=("scratch_size_bytes", "first"),
            estimated_total_bytes=("estimated_total_bytes", "first"),
            implementation_order=("implementation_order", "first"),
        )
        .sort_values(["valid_ratio_percent", "implementation_order"])
        .reset_index(drop=True)
    )

    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    summary["p95_to_median_gpu_ms"] = summary["gpu_ms_p95"] / summary["gpu_ms_median"]
    summary["p95_to_median_gbps"] = summary["gbps_p95"] / summary["gbps_median"]
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    atomic_rows = (
        summary[summary["implementation"] == "global_atomic_append"][
            ["valid_ratio_percent", "logical_elements", "gpu_ms_median", "gbps_median", "throughput_median"]
        ]
        .rename(
            columns={
                "gpu_ms_median": "atomic_gpu_ms_median",
                "gbps_median": "atomic_gbps_median",
                "throughput_median": "atomic_throughput_median",
            }
        )
        .reset_index(drop=True)
    )
    three_stage_rows = (
        summary[summary["implementation"] == "three_stage"][
            ["valid_ratio_percent", "logical_elements", "gpu_ms_median", "gbps_median", "throughput_median"]
        ]
        .rename(
            columns={
                "gpu_ms_median": "three_stage_gpu_ms_median",
                "gbps_median": "three_stage_gbps_median",
                "throughput_median": "three_stage_throughput_median",
            }
        )
        .reset_index(drop=True)
    )

    relative = atomic_rows.merge(three_stage_rows, on=["valid_ratio_percent", "logical_elements"], how="inner")
    if relative.empty:
        raise ValueError("Could not build three-stage-versus-atomic relative table.")

    relative["three_stage_speedup_vs_atomic"] = (
        relative["atomic_gpu_ms_median"] / relative["three_stage_gpu_ms_median"]
    )
    relative["three_stage_gpu_delta_pct_vs_atomic"] = (
        (relative["three_stage_gpu_ms_median"] - relative["atomic_gpu_ms_median"]) / relative["atomic_gpu_ms_median"]
    ) * 100.0
    relative["three_stage_gbps_ratio_vs_atomic"] = (
        relative["three_stage_gbps_median"] / relative["atomic_gbps_median"]
    )
    relative["three_stage_throughput_ratio_vs_atomic"] = (
        relative["three_stage_throughput_median"] / relative["atomic_throughput_median"]
    )
    return relative.sort_values(["valid_ratio_percent"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        [
            "implementation",
            "valid_ratio_percent",
            "logical_elements",
            "gpu_ms_p95",
            "p95_to_median_gpu_ms",
            "gpu_ms_cv",
            "gbps_p95",
            "p95_to_median_gbps",
            "gbps_cv",
            "sample_count_rows",
            "correctness_pass_rate",
        ]
    ].sort_values(["valid_ratio_percent", "implementation"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 24 stream compaction data.")
    parser.add_argument("--skip-current", action="store_true")
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current)

    summary = _build_summary(frame)
    relative = _build_relative(summary)
    stability = _build_stability(summary)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _write_table(summary, TABLES_DIR / "stream_compaction_summary.csv")
    _write_table(relative, TABLES_DIR / "stream_compaction_relative.csv")
    _write_table(stability, TABLES_DIR / "stream_compaction_stability.csv")

    rows = int(frame.shape[0])
    cases = int(summary.shape[0])
    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(f"[ok] Wrote Experiment 24 analysis tables from {source_label} ({rows} rows, {cases} cases on {gpu_name}).")


if __name__ == "__main__":
    main()
