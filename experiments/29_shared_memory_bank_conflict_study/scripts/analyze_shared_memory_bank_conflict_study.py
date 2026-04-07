#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "29_shared_memory_bank_conflict_study"

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
        raise ValueError("Input benchmark JSON has no Experiment 29 rows.")

    for column in ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
    note_maps = frame["notes"].fillna("").map(_parse_notes)
    frame["shared_stride_elements"] = pd.to_numeric(
        note_maps.map(lambda notes: notes.get("shared_stride_elements", "0")), errors="coerce"
    )
    frame["padding_fix"] = note_maps.map(lambda notes: notes.get("padding_fix", "false")).astype(str)
    return frame, payload.get("metadata", {})


def _load_source_frame(skip_current: bool) -> tuple[pd.DataFrame, dict, Path]:
    if not skip_current and CURRENT_JSON.exists():
        frame, metadata = _load_frame(CURRENT_JSON)
        return frame, metadata, CURRENT_JSON
    latest = _latest_run_path()
    if latest is None:
        raise FileNotFoundError("No benchmark_results.json or archived run JSON found for Experiment 29.")
    frame, metadata = _load_frame(latest)
    return frame, metadata, latest


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["variant", "shared_stride_elements", "problem_size"], as_index=False)
        .agg(
            sample_count=("gpu_ms", "count"),
            correctness_pass_rate=("correctness_pass", "mean"),
            gpu_ms_median=("gpu_ms", "median"),
            gpu_ms_p95=("gpu_ms", _quantile_95),
            gbps_median=("gbps", "median"),
            throughput_median=("throughput", "median"),
            padding_fix=("padding_fix", "first"),
        )
        .sort_values(["shared_stride_elements", "variant"])
        .reset_index(drop=True)
    )


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["variant"] == "stride_1"][["problem_size", "gpu_ms_median", "gbps_median"]].rename(
        columns={"gpu_ms_median": "baseline_gpu_ms_median", "gbps_median": "baseline_gbps_median"}
    )
    relative = summary.merge(baseline, on="problem_size", how="inner")
    relative["slowdown_vs_stride_1"] = relative["gpu_ms_median"] / relative["baseline_gpu_ms_median"]
    relative["gbps_ratio_vs_stride_1"] = relative["gbps_median"] / relative["baseline_gbps_median"]
    return relative[
        [
            "variant",
            "shared_stride_elements",
            "problem_size",
            "gpu_ms_median",
            "slowdown_vs_stride_1",
            "gbps_median",
            "gbps_ratio_vs_stride_1",
            "padding_fix",
        ]
    ]


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[["variant", "shared_stride_elements", "problem_size", "gpu_ms_median", "gpu_ms_p95"]].copy()
    stability["p95_to_median_gpu_ms"] = stability["gpu_ms_p95"] / stability["gpu_ms_median"]
    return stability


def _write_table(frame: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / name, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 29 bank-conflict data.")
    parser.add_argument("--skip-current", action="store_true")
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current)
    summary = _build_summary(frame)
    relative = _build_relative(summary)
    stability = _build_stability(summary)
    _write_table(summary, "shared_memory_bank_conflict_study_summary.csv")
    _write_table(relative, "shared_memory_bank_conflict_study_relative.csv")
    _write_table(stability, "shared_memory_bank_conflict_study_stability.csv")
    print(
        f"[ok] Wrote Experiment 29 analysis tables from {source_path.name} on "
        f"{metadata.get('gpu_name', 'unknown_gpu')}."
    )


if __name__ == "__main__":
    main()
