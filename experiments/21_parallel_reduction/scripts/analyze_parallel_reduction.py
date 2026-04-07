#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "21_parallel_reduction"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"

VARIANT_ORDER = {
    "global_atomic": 0,
    "shared_tree": 1,
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

    frame = frame[frame["experiment_id"] == EXPERIMENT_ID].copy()
    if frame.empty:
        raise ValueError("Input benchmark JSON has rows but no Experiment 21 entries.")

    numeric_columns = ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "dispatch_count" in frame.columns:
        frame["dispatch_count"] = pd.to_numeric(frame["dispatch_count"], errors="coerce")
    else:
        frame["dispatch_count"] = 1

    frame["variant"] = frame["variant"].fillna("unknown_variant").astype(str)
    frame["notes"] = frame["notes"].fillna("").astype(str) if "notes" in frame.columns else ""
    frame["correctness_pass"] = (
        frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
        if "correctness_pass" in frame.columns
        else False
    )

    note_maps = frame["notes"].apply(_parse_notes_field)
    for column_name, note_key in [
        ("reduction_strategy", "reduction_strategy"),
        ("workgroup_size", "workgroup_size"),
        ("staged_reduction_levels", "staged_reduction_levels"),
        ("group_count_x", "group_count_x"),
        ("input_elements", "input_elements"),
        ("input_bytes", "input_bytes"),
        ("output_bytes", "output_bytes"),
        ("logical_bytes_touched", "logical_bytes_touched"),
        ("estimated_atomic_bytes", "estimated_atomic_bytes"),
        ("shared_bytes_per_workgroup", "shared_bytes_per_workgroup"),
    ]:
        frame[column_name] = pd.to_numeric(note_maps.apply(lambda mapping: mapping.get(note_key)), errors="coerce")

    frame["reduction_strategy"] = frame["reduction_strategy"].fillna(frame["variant"])
    frame["workgroup_size"] = frame["workgroup_size"].fillna(256)
    frame["staged_reduction_levels"] = frame["staged_reduction_levels"].fillna(0)
    frame["group_count_x"] = frame["group_count_x"].fillna(0)
    frame["input_elements"] = frame["input_elements"].fillna(frame["problem_size"])
    frame["input_bytes"] = frame["input_bytes"].fillna(frame["problem_size"] * 4)
    frame["output_bytes"] = frame["output_bytes"].fillna(4)
    frame["logical_bytes_touched"] = frame["logical_bytes_touched"].fillna(frame["input_bytes"] + frame["output_bytes"])
    frame["estimated_atomic_bytes"] = frame["estimated_atomic_bytes"].fillna(frame["output_bytes"])
    frame["shared_bytes_per_workgroup"] = frame["shared_bytes_per_workgroup"].fillna(0)

    for column_name in [
        "workgroup_size",
        "staged_reduction_levels",
        "group_count_x",
        "input_elements",
        "input_bytes",
        "output_bytes",
        "logical_bytes_touched",
        "estimated_atomic_bytes",
        "shared_bytes_per_workgroup",
    ]:
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
        if frame[column_name].isna().any():
            raise ValueError(f"Could not determine {column_name} for one or more rows.")
        frame[column_name] = frame[column_name].astype(int)

    frame["problem_size"] = frame["problem_size"].astype(int)
    frame["dispatch_count"] = frame["dispatch_count"].fillna(1).astype(int)
    frame["iteration"] = frame["iteration"].fillna(0).astype(int)
    frame["variant_order"] = frame["variant"].map(lambda value: VARIANT_ORDER.get(value, len(VARIANT_ORDER)))
    frame["input_mib"] = frame["input_bytes"] / (1024.0 * 1024.0)

    metadata = payload.get("metadata", {})
    return frame, metadata


def _load_source_frame(skip_current: bool, runs_dir: Path) -> tuple[pd.DataFrame, dict, Path]:
    if not skip_current and CURRENT_JSON.exists():
        frame, metadata = _load_frame(CURRENT_JSON)
        return frame, metadata, CURRENT_JSON

    latest_run = _latest_run_path(runs_dir)
    if latest_run is None:
        if CURRENT_JSON.exists():
            frame, metadata = _load_frame(CURRENT_JSON)
            return frame, metadata, CURRENT_JSON
        raise FileNotFoundError(
            "No benchmark_results.json or archived run JSON found for Experiment 21. Run data collection first."
        )

    frame, metadata = _load_frame(latest_run)
    return frame, metadata, latest_run


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["variant", "problem_size"], as_index=False)
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
            workgroup_size=("workgroup_size", "first"),
            staged_reduction_levels=("staged_reduction_levels", "first"),
            group_count_x=("group_count_x", "first"),
            input_elements=("input_elements", "first"),
            input_bytes=("input_bytes", "first"),
            output_bytes=("output_bytes", "first"),
            logical_bytes_touched=("logical_bytes_touched", "first"),
            estimated_atomic_bytes=("estimated_atomic_bytes", "first"),
            shared_bytes_per_workgroup=("shared_bytes_per_workgroup", "first"),
            reduction_strategy=("reduction_strategy", "first"),
            variant_order=("variant_order", "first"),
        )
        .sort_values(["problem_size", "variant_order", "variant"])
        .reset_index(drop=True)
    )

    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    summary["input_mib"] = summary["input_bytes"] / (1024.0 * 1024.0)
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        summary[summary["variant"] == "global_atomic"][
            ["problem_size", "input_mib", "gpu_ms_median", "gbps_median", "throughput_median"]
        ]
        .rename(
            columns={
                "gpu_ms_median": "global_atomic_gpu_ms_median",
                "gbps_median": "global_atomic_gbps_median",
                "throughput_median": "global_atomic_throughput_median",
            }
        )
        .reset_index(drop=True)
    )
    shared = (
        summary[summary["variant"] == "shared_tree"][
            ["problem_size", "input_mib", "gpu_ms_median", "gbps_median", "throughput_median"]
        ]
        .rename(
            columns={
                "gpu_ms_median": "shared_tree_gpu_ms_median",
                "gbps_median": "shared_tree_gbps_median",
                "throughput_median": "shared_tree_throughput_median",
            }
        )
        .reset_index(drop=True)
    )

    relative = baseline.merge(shared, on=["problem_size", "input_mib"], how="inner")
    if relative.empty:
        raise ValueError("Could not build global_atomic-versus-shared_tree relative table.")

    relative["shared_tree_speedup_vs_global_atomic"] = (
        relative["global_atomic_gpu_ms_median"] / relative["shared_tree_gpu_ms_median"]
    )
    relative["shared_tree_gpu_delta_pct_vs_global_atomic"] = (
        (relative["shared_tree_gpu_ms_median"] - relative["global_atomic_gpu_ms_median"])
        / relative["global_atomic_gpu_ms_median"]
    ) * 100.0
    relative["shared_tree_gbps_ratio_vs_global_atomic"] = (
        relative["shared_tree_gbps_median"] / relative["global_atomic_gbps_median"]
    )
    relative["shared_tree_throughput_ratio_vs_global_atomic"] = (
        relative["shared_tree_throughput_median"] / relative["global_atomic_throughput_median"]
    )
    return relative.sort_values(["problem_size"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "variant",
            "problem_size",
            "input_mib",
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
    return stability.sort_values(["problem_size", "variant"]).reset_index(drop=True)


def _build_status_overview(latest_rows: pd.DataFrame) -> pd.DataFrame:
    total_rows = int(latest_rows.shape[0])
    pass_rows = int(latest_rows["correctness_pass"].astype(bool).sum())
    return pd.DataFrame(
        [
            {
                "total_rows": total_rows,
                "correctness_pass_count": pass_rows,
                "correctness_fail_count": total_rows - pass_rows,
                "correctness_pass_rate": float(pass_rows / total_rows) if total_rows > 0 else 0.0,
            }
        ]
    )


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 21 parallel reduction data.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing collected run JSON files.",
    )
    parser.add_argument(
        "--skip-current",
        action="store_true",
        help="Prefer the latest archived run under runs/ instead of results/tables/benchmark_results.json.",
    )
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current, runs_dir=args.runs_dir)
    summary = _build_summary(frame)
    relative = _build_relative(summary)
    stability = _build_stability(summary)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _write_table(summary, TABLES_DIR / "parallel_reduction_summary.csv")
    _write_table(relative, TABLES_DIR / "parallel_reduction_relative.csv")
    _write_table(stability, TABLES_DIR / "parallel_reduction_stability.csv")
    _write_table(_build_status_overview(frame), TABLES_DIR / "parallel_reduction_status_overview.csv")

    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(f"[ok] Wrote Experiment 21 analysis tables from {source_label} ({frame.shape[0]} rows on {gpu_name}).")


if __name__ == "__main__":
    main()
