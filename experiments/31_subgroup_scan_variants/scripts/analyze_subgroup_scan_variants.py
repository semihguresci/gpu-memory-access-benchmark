#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analysis_integrity import (
    latest_compatible_run,
    select_compatible_runs,
    validated_rows,
    write_analysis_provenance,
)

EXPERIMENT_ID = "31_subgroup_scan_variants"

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
    return latest_compatible_run(candidates)


def _load_frame(path: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(validated_rows(payload, EXPERIMENT_ID, path))
    if frame.empty:
        raise ValueError("Input benchmark JSON has no rows[].")
    frame = frame[frame["experiment_id"] == EXPERIMENT_ID].copy()
    if frame.empty:
        raise ValueError("Input benchmark JSON has no Experiment 31 rows.")
    for column in ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
    note_maps = frame["notes"].fillna("").map(_parse_notes)
    frame["items_per_thread"] = pd.to_numeric(
        note_maps.map(lambda notes: notes.get("items_per_thread", "1")), errors="coerce"
    )
    return frame, payload.get("metadata", {})


def _load_source_frame(skip_current: bool) -> tuple[pd.DataFrame, dict, Path]:
    if not skip_current and CURRENT_JSON.exists():
        frame, metadata = _load_frame(CURRENT_JSON)
        return frame, metadata, CURRENT_JSON
    latest = _latest_run_path()
    if latest is None:
        raise FileNotFoundError("No benchmark_results.json or archived run JSON found for Experiment 31.")
    frame, metadata = _load_frame(latest)
    return frame, metadata, latest


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["variant", "items_per_thread", "problem_size"], as_index=False)
        .agg(
            sample_count=("gpu_ms", "count"),
            correctness_pass_rate=("correctness_pass", "mean"),
            gpu_ms_median=("gpu_ms", "median"),
            gpu_ms_p95=("gpu_ms", _quantile_95),
            gbps_median=("gbps", "median"),
            throughput_median=("throughput", "median"),
        )
        .sort_values(["items_per_thread", "variant"])
        .reset_index(drop=True)
    )


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["variant"].str.startswith("shared_block_scan")][
        ["items_per_thread", "problem_size", "gpu_ms_median", "gbps_median"]
    ].rename(columns={"gpu_ms_median": "baseline_gpu_ms_median", "gbps_median": "baseline_gbps_median"})
    relative = summary.merge(baseline, on=["items_per_thread", "problem_size"], how="inner")
    relative["speedup_vs_shared_block_scan"] = relative["baseline_gpu_ms_median"] / relative["gpu_ms_median"]
    relative["gbps_ratio_vs_shared_block_scan"] = relative["gbps_median"] / relative["baseline_gbps_median"]
    return relative[
        [
            "variant",
            "items_per_thread",
            "problem_size",
            "gpu_ms_median",
            "speedup_vs_shared_block_scan",
            "gbps_median",
            "gbps_ratio_vs_shared_block_scan",
        ]
    ]


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[["variant", "items_per_thread", "problem_size", "gpu_ms_median", "gpu_ms_p95"]].copy()
    stability["p95_to_median_gpu_ms"] = stability["gpu_ms_p95"] / stability["gpu_ms_median"]
    return stability


def _write_table(frame: pd.DataFrame, name: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / name, index=False, float_format="%.6f")
    return TABLES_DIR / name


def main() -> None:
    generated_tables: list[Path] = []
    generated_charts: list[Path | None] = []
    parser = argparse.ArgumentParser(description="Analyze Experiment 31 subgroup scan data.")
    parser.add_argument("--skip-current", action="store_true")
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current)
    summary = _build_summary(frame)
    relative = _build_relative(summary)
    stability = _build_stability(summary)
    generated_tables.append(_write_table(summary, "subgroup_scan_variants_summary.csv"))
    generated_tables.append(_write_table(relative, "subgroup_scan_variants_relative.csv"))
    generated_tables.append(_write_table(stability, "subgroup_scan_variants_stability.csv"))
    write_analysis_provenance(TABLES_DIR, [source_path], EXPERIMENT_ID, derived_tables=generated_tables, derived_charts=generated_charts, analysis_tools=[Path(__file__)])
    print(
        f"[ok] Wrote Experiment 31 analysis tables from {source_path.name} on "
        f"{metadata.get('gpu_name', 'unknown_gpu')}."
    )


if __name__ == "__main__":
    main()
