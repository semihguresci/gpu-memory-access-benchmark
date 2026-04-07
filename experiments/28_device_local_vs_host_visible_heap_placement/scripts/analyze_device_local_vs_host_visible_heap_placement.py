#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "28_device_local_vs_host_visible_heap_placement"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"


def _parse_notes(notes: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for token in str(notes).split(";"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        pairs[key.strip()] = value.strip()
    return pairs


def _latest_run_path() -> Path | None:
    candidates = sorted(path for path in RUNS_DIR.rglob("*.json") if path.is_file()) if RUNS_DIR.exists() else []
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name)) if candidates else None


def _load_frame(path: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload.get("rows", []))
    if frame.empty:
        raise ValueError("Input benchmark JSON has no rows[].")
    frame = frame[frame["experiment_id"] == EXPERIMENT_ID].copy()
    if frame.empty:
        raise ValueError("Input benchmark JSON has no Experiment 28 rows.")

    for column in ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
    note_maps = frame["notes"].fillna("").map(_parse_notes)
    frame["placement"] = note_maps.map(lambda notes: notes.get("placement", "unknown"))
    frame["upload_ms"] = pd.to_numeric(note_maps.map(lambda notes: notes.get("upload_ms", "0")), errors="coerce")
    frame["readback_ms"] = pd.to_numeric(note_maps.map(lambda notes: notes.get("readback_ms", "0")), errors="coerce")
    frame["resident_buffer_count"] = pd.to_numeric(
        note_maps.map(lambda notes: notes.get("resident_buffer_count", "0")), errors="coerce"
    )
    return frame, payload.get("metadata", {})


def _load_source_frame(skip_current: bool) -> tuple[pd.DataFrame, dict, Path]:
    if not skip_current and CURRENT_JSON.exists():
        frame, metadata = _load_frame(CURRENT_JSON)
        return frame, metadata, CURRENT_JSON
    latest = _latest_run_path()
    if latest is None:
        raise FileNotFoundError("No benchmark_results.json or archived run JSON found for Experiment 28.")
    frame, metadata = _load_frame(latest)
    return frame, metadata, latest


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["variant", "placement", "problem_size"], as_index=False)
        .agg(
            sample_count=("gpu_ms", "count"),
            correctness_pass_rate=("correctness_pass", "mean"),
            gpu_ms_mean=("gpu_ms", "mean"),
            gpu_ms_median=("gpu_ms", "median"),
            gpu_ms_p95=("gpu_ms", _quantile_95),
            end_to_end_ms_median=("end_to_end_ms", "median"),
            throughput_median=("throughput", "median"),
            gbps_median=("gbps", "median"),
            upload_ms_median=("upload_ms", "median"),
            readback_ms_median=("readback_ms", "median"),
            resident_buffer_count=("resident_buffer_count", "first"),
        )
        .sort_values(["problem_size", "variant"])
        .reset_index(drop=True)
    )
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["variant"] == "host_visible_direct"][
        ["problem_size", "gpu_ms_median", "end_to_end_ms_median", "gbps_median"]
    ].rename(
        columns={
            "gpu_ms_median": "baseline_gpu_ms_median",
            "end_to_end_ms_median": "baseline_end_to_end_ms_median",
            "gbps_median": "baseline_gbps_median",
        }
    )
    relative = summary.merge(baseline, on="problem_size", how="inner")
    relative["slowdown_vs_host_visible_dispatch"] = relative["gpu_ms_median"] / relative["baseline_gpu_ms_median"]
    relative["slowdown_vs_host_visible_end_to_end"] = (
        relative["end_to_end_ms_median"] / relative["baseline_end_to_end_ms_median"]
    )
    relative["dispatch_gbps_ratio_vs_host_visible"] = relative["gbps_median"] / relative["baseline_gbps_median"]
    return relative[
        [
            "variant",
            "problem_size",
            "gpu_ms_median",
            "end_to_end_ms_median",
            "slowdown_vs_host_visible_dispatch",
            "slowdown_vs_host_visible_end_to_end",
            "gbps_median",
            "dispatch_gbps_ratio_vs_host_visible",
            "upload_ms_median",
            "readback_ms_median",
        ]
    ]


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "variant",
            "problem_size",
            "gpu_ms_median",
            "gpu_ms_p95",
            "sample_count",
            "correctness_pass_rate",
        ]
    ].copy()
    stability["p95_to_median_gpu_ms"] = stability["gpu_ms_p95"] / stability["gpu_ms_median"]
    return stability


def _write_table(frame: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / name, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 28 heap placement data.")
    parser.add_argument("--skip-current", action="store_true")
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current)
    summary = _build_summary(frame)
    relative = _build_relative(summary)
    stability = _build_stability(summary)
    _write_table(summary, "device_local_vs_host_visible_heap_placement_summary.csv")
    _write_table(relative, "device_local_vs_host_visible_heap_placement_relative.csv")
    _write_table(stability, "device_local_vs_host_visible_heap_placement_stability.csv")
    print(
        f"[ok] Wrote Experiment 28 analysis tables from {source_path.name} on "
        f"{metadata.get('gpu_name', 'unknown_gpu')}."
    )


if __name__ == "__main__":
    main()
