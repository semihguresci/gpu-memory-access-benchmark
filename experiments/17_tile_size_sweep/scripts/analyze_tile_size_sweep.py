#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "17_tile_size_sweep"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"

IMPLEMENTATION_ORDER = {
    "direct_global": 0,
    "shared_tiled": 1,
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
        raise ValueError("Input benchmark JSON has rows but no Experiment 17 entries.")

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
        ("implementation", "implementation"),
        ("tile_size", "tile_size"),
        ("radius", "radius"),
        ("center_offset", "center_offset"),
        ("local_size_x", "local_size_x"),
        ("group_count_x", "group_count_x"),
        ("tile_span_elements", "tile_span_elements"),
        ("load_rounds", "load_rounds"),
        ("shared_bytes_per_workgroup", "shared_bytes_per_workgroup"),
        ("barriers_per_workgroup", "barriers_per_workgroup"),
        ("estimated_global_read_bytes", "estimated_global_read_bytes"),
        ("estimated_global_write_bytes", "estimated_global_write_bytes"),
    ]:
        frame[column_name] = note_maps.apply(lambda mapping: mapping.get(note_key))

    frame["implementation"] = frame["implementation"].fillna(
        frame["variant"].str.extract(r"^(direct_global|shared_tiled)")[0]
    )
    frame["tile_size"] = pd.to_numeric(
        frame["tile_size"].fillna(frame["variant"].str.extract(r"_t(\d+)$")[0]), errors="coerce"
    )

    for column_name, default_value in [
        ("radius", 16),
        ("center_offset", 16),
        ("local_size_x", 0),
        ("group_count_x", 0),
        ("tile_span_elements", 0),
        ("load_rounds", 0),
        ("shared_bytes_per_workgroup", 0),
        ("barriers_per_workgroup", 0),
        ("estimated_global_read_bytes", 0),
        ("estimated_global_write_bytes", 0),
    ]:
        frame[column_name] = pd.to_numeric(frame[column_name].fillna(default_value), errors="coerce")
        if frame[column_name].isna().any():
            raise ValueError(f"Could not determine {column_name} for one or more rows.")

    if frame["tile_size"].isna().any():
        raise ValueError("Could not determine tile_size for one or more rows.")

    frame["tile_size"] = frame["tile_size"].astype(int)
    frame["radius"] = frame["radius"].astype(int)
    frame["center_offset"] = frame["center_offset"].astype(int)
    frame["local_size_x"] = frame["local_size_x"].astype(int)
    frame["group_count_x"] = frame["group_count_x"].astype(int)
    frame["tile_span_elements"] = frame["tile_span_elements"].astype(int)
    frame["load_rounds"] = frame["load_rounds"].astype(int)
    frame["shared_bytes_per_workgroup"] = frame["shared_bytes_per_workgroup"].astype(int)
    frame["barriers_per_workgroup"] = frame["barriers_per_workgroup"].astype(int)
    frame["estimated_global_read_bytes"] = frame["estimated_global_read_bytes"].astype(int)
    frame["estimated_global_write_bytes"] = frame["estimated_global_write_bytes"].astype(int)
    frame["logical_elements"] = frame["problem_size"].astype(int)
    frame["implementation_order"] = frame["implementation"].map(
        lambda value: IMPLEMENTATION_ORDER.get(str(value), len(IMPLEMENTATION_ORDER))
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
            "No benchmark_results.json or archived run JSON found for Experiment 17. Run data collection first."
        )

    frame, metadata = _load_frame(latest_run)
    return frame, metadata, latest_run


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["implementation", "tile_size", "logical_elements"], as_index=False)
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
            radius=("radius", "first"),
            center_offset=("center_offset", "first"),
            local_size_x=("local_size_x", "first"),
            group_count_x=("group_count_x", "first"),
            tile_span_elements=("tile_span_elements", "first"),
            load_rounds=("load_rounds", "first"),
            shared_bytes_per_workgroup=("shared_bytes_per_workgroup", "first"),
            barriers_per_workgroup=("barriers_per_workgroup", "first"),
            estimated_global_read_bytes=("estimated_global_read_bytes", "first"),
            estimated_global_write_bytes=("estimated_global_write_bytes", "first"),
            implementation_order=("implementation_order", "first"),
        )
        .sort_values(["tile_size", "implementation_order", "implementation"])
        .reset_index(drop=True)
    )

    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    summary["p95_to_median_gpu_ms"] = summary["gpu_ms_p95"] / summary["gpu_ms_median"]
    summary["p95_to_median_gbps"] = summary["gbps_p95"] / summary["gbps_median"]
    summary["estimated_global_total_bytes"] = (
        summary["estimated_global_read_bytes"] + summary["estimated_global_write_bytes"]
    )
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    direct = (
        summary[summary["implementation"] == "direct_global"][
            ["tile_size", "logical_elements", "gpu_ms_median", "gbps_median", "throughput_median"]
        ]
        .rename(
            columns={
                "gpu_ms_median": "direct_gpu_ms_median",
                "gbps_median": "direct_gbps_median",
                "throughput_median": "direct_throughput_median",
            }
        )
        .reset_index(drop=True)
    )
    tiled = (
        summary[summary["implementation"] == "shared_tiled"][
            ["tile_size", "logical_elements", "gpu_ms_median", "gbps_median", "throughput_median"]
        ]
        .rename(
            columns={
                "gpu_ms_median": "tiled_gpu_ms_median",
                "gbps_median": "tiled_gbps_median",
                "throughput_median": "tiled_throughput_median",
            }
        )
        .reset_index(drop=True)
    )

    relative = direct.merge(tiled, on=["tile_size", "logical_elements"], how="inner")
    if relative.empty:
        raise ValueError("Could not build direct-versus-tiled relative table.")

    relative["tiled_speedup_vs_direct"] = relative["direct_gpu_ms_median"] / relative["tiled_gpu_ms_median"]
    relative["tiled_gpu_delta_pct_vs_direct"] = (
        (relative["tiled_gpu_ms_median"] - relative["direct_gpu_ms_median"]) / relative["direct_gpu_ms_median"]
    ) * 100.0
    relative["tiled_gbps_ratio_vs_direct"] = relative["tiled_gbps_median"] / relative["direct_gbps_median"]
    relative["tiled_throughput_ratio_vs_direct"] = (
        relative["tiled_throughput_median"] / relative["direct_throughput_median"]
    )
    return relative.sort_values(["tile_size"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "implementation",
            "tile_size",
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
    return stability.sort_values(["tile_size", "implementation"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 17 tile size sweep data.")
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
    _write_table(summary, TABLES_DIR / "tile_size_sweep_summary.csv")
    _write_table(relative, TABLES_DIR / "tile_size_sweep_relative.csv")
    _write_table(stability, TABLES_DIR / "tile_size_sweep_stability.csv")

    rows = int(frame.shape[0])
    variants = int(summary.shape[0])
    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(f"[ok] Wrote Experiment 17 analysis tables from {source_label} ({rows} rows, {variants} cases on {gpu_name}).")


if __name__ == "__main__":
    main()
