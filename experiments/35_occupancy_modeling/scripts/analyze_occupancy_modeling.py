#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "35_occupancy_modeling"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"


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
        raise ValueError(f"Input benchmark JSON has no {EXPERIMENT_ID} rows.")
    for column in ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
    return frame, payload.get("metadata", {})


def _load_source_frame(skip_current: bool) -> tuple[pd.DataFrame, dict, Path]:
    if not skip_current and CURRENT_JSON.exists():
        frame, metadata = _load_frame(CURRENT_JSON)
        return frame, metadata, CURRENT_JSON
    latest = _latest_run_path()
    if latest is None:
        raise FileNotFoundError(f"No benchmark_results.json or archived run JSON found for {EXPERIMENT_ID}.")
    frame, metadata = _load_frame(latest)
    return frame, metadata, latest


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["variant", "problem_size"], as_index=False)
        .agg(
            sample_count=("gpu_ms", "count"),
            correctness_pass_rate=("correctness_pass", "mean"),
            gpu_ms_median=("gpu_ms", "median"),
            gpu_ms_p95=("gpu_ms", _quantile_95),
            throughput_median=("throughput", "median"),
            gbps_median=("gbps", "median"),
        )
        .sort_values(["problem_size", "variant"])
        .reset_index(drop=True)
    )


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["variant"] == "low_smem"][
        ["problem_size", "gpu_ms_median", "gbps_median"]
    ].rename(columns={"gpu_ms_median": "baseline_gpu_ms_median", "gbps_median": "baseline_gbps_median"})
    relative = summary.merge(baseline, on="problem_size", how="inner")
    relative["slowdown_vs_low_smem"] = relative["gpu_ms_median"] / relative["baseline_gpu_ms_median"]
    relative["gbps_ratio_vs_low_smem"] = relative["gbps_median"] / relative["baseline_gbps_median"]
    return relative[
        [
            "variant",
            "problem_size",
            "gpu_ms_median",
            "slowdown_vs_low_smem",
            "gbps_median",
            "gbps_ratio_vs_low_smem",
        ]
    ]


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[["variant", "problem_size", "gpu_ms_median", "gpu_ms_p95"]].copy()
    stability["p95_to_median_gpu_ms"] = stability["gpu_ms_p95"] / stability["gpu_ms_median"]
    return stability


def _write_table(frame: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / name, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Analyze {EXPERIMENT_ID} data.")
    parser.add_argument("--skip-current", action="store_true")
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current)
    summary = _build_summary(frame)
    relative = _build_relative(summary)
    stability = _build_stability(summary)
    _write_table(summary, "occupancy_modeling_summary.csv")
    _write_table(relative, "occupancy_modeling_relative.csv")
    _write_table(stability, "occupancy_modeling_stability.csv")
    print(
        f"[ok] Wrote {EXPERIMENT_ID} analysis tables from {source_path.name} on "
        f"{metadata.get('gpu_name', 'unknown_gpu')}."
    )


if __name__ == "__main__":
    main()
