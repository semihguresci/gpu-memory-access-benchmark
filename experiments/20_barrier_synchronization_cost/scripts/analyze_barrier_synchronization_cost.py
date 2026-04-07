#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "20_barrier_synchronization_cost"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"

PLACEMENT_ORDER = {
    "flat_loop": 0,
    "tiled_regions": 1,
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
        raise ValueError("Input benchmark JSON has rows but no Experiment 20 entries.")

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
        ("placement", "placement"),
        ("barrier_interval_phases", "barrier_interval_phases"),
        ("sync_points", "sync_points"),
        ("work_phases", "work_phases"),
        ("tile_phase_count", "tile_phase_count"),
        ("tile_count", "tile_count"),
        ("local_size_x", "local_size_x"),
        ("group_count_x", "group_count_x"),
        ("shared_bytes_per_workgroup", "shared_bytes_per_workgroup"),
        ("logical_elements", "logical_elements"),
    ]:
        frame[column_name] = note_maps.apply(lambda mapping: mapping.get(note_key))

    frame["placement"] = frame["placement"].fillna(frame["variant"].str.extract(r"^(flat_loop|tiled_regions)")[0])
    for column_name, default_value in [
        ("barrier_interval_phases", 0),
        ("sync_points", 0),
        ("work_phases", 8),
        ("tile_phase_count", 0),
        ("tile_count", 0),
        ("local_size_x", 256),
        ("group_count_x", 0),
        ("shared_bytes_per_workgroup", 0),
    ]:
        frame[column_name] = pd.to_numeric(frame[column_name].fillna(default_value), errors="coerce")
        if frame[column_name].isna().any():
            raise ValueError(f"Could not determine {column_name} for one or more rows.")

    frame["logical_elements"] = pd.to_numeric(frame["logical_elements"].fillna(frame["problem_size"]), errors="coerce")
    if frame["logical_elements"].isna().any():
        raise ValueError("Could not determine logical_elements for one or more rows.")

    frame["barrier_interval_phases"] = frame["barrier_interval_phases"].astype(int)
    frame["sync_points"] = frame["sync_points"].astype(int)
    frame["work_phases"] = frame["work_phases"].astype(int)
    frame["tile_phase_count"] = frame["tile_phase_count"].astype(int)
    frame["tile_count"] = frame["tile_count"].astype(int)
    frame["local_size_x"] = frame["local_size_x"].astype(int)
    frame["group_count_x"] = frame["group_count_x"].astype(int)
    frame["shared_bytes_per_workgroup"] = frame["shared_bytes_per_workgroup"].astype(int)
    frame["logical_elements"] = frame["logical_elements"].astype(int)

    frame["placement_order"] = frame["placement"].map(lambda value: PLACEMENT_ORDER.get(str(value), len(PLACEMENT_ORDER)))

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
            "No benchmark_results.json or archived run JSON found for Experiment 20. "
            "Run data collection first."
        )

    frame, metadata = _load_frame(latest_run)
    return frame, metadata, latest_run


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["placement", "barrier_interval_phases", "logical_elements"], as_index=False)
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
            sync_points=("sync_points", "first"),
            work_phases=("work_phases", "first"),
            tile_phase_count=("tile_phase_count", "first"),
            tile_count=("tile_count", "first"),
            local_size_x=("local_size_x", "first"),
            group_count_x=("group_count_x", "first"),
            shared_bytes_per_workgroup=("shared_bytes_per_workgroup", "first"),
            placement_order=("placement_order", "first"),
        )
        .sort_values(["barrier_interval_phases", "placement_order", "placement"])
        .reset_index(drop=True)
    )

    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    summary["p95_to_median_gpu_ms"] = summary["gpu_ms_p95"] / summary["gpu_ms_median"]
    summary["p95_to_median_gbps"] = summary["gbps_p95"] / summary["gbps_median"]
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    flat = (
        summary[summary["placement"] == "flat_loop"][
            [
                "barrier_interval_phases",
                "logical_elements",
                "gpu_ms_median",
                "gbps_median",
                "throughput_median",
            ]
        ]
        .rename(
            columns={
                "gpu_ms_median": "flat_gpu_ms_median",
                "gbps_median": "flat_gbps_median",
                "throughput_median": "flat_throughput_median",
            }
        )
        .reset_index(drop=True)
    )
    tiled = (
        summary[summary["placement"] == "tiled_regions"][
            [
                "barrier_interval_phases",
                "logical_elements",
                "gpu_ms_median",
                "gbps_median",
                "throughput_median",
            ]
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

    relative = flat.merge(tiled, on=["barrier_interval_phases", "logical_elements"], how="inner")
    if relative.empty:
        raise ValueError("Could not build flat-versus-tiled relative table.")

    relative["tiled_speedup_vs_flat"] = relative["flat_gpu_ms_median"] / relative["tiled_gpu_ms_median"]
    relative["tiled_gpu_delta_pct_vs_flat"] = (
        (relative["tiled_gpu_ms_median"] - relative["flat_gpu_ms_median"]) / relative["flat_gpu_ms_median"]
    ) * 100.0
    relative["tiled_gbps_ratio_vs_flat"] = relative["tiled_gbps_median"] / relative["flat_gbps_median"]
    relative["tiled_throughput_ratio_vs_flat"] = (
        relative["tiled_throughput_median"] / relative["flat_throughput_median"]
    )
    return relative.sort_values(["barrier_interval_phases"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "placement",
            "barrier_interval_phases",
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
    return stability.sort_values(["barrier_interval_phases", "placement"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 20 barrier synchronization cost data.")
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
    _write_table(summary, TABLES_DIR / "barrier_synchronization_cost_summary.csv")
    _write_table(relative, TABLES_DIR / "barrier_synchronization_cost_relative.csv")
    _write_table(stability, TABLES_DIR / "barrier_synchronization_cost_stability.csv")

    rows = int(frame.shape[0])
    variants = int(summary.shape[0])
    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(f"[ok] Wrote Experiment 20 analysis tables from {source_label} ({rows} rows, {variants} cases on {gpu_name}).")


if __name__ == "__main__":
    main()
