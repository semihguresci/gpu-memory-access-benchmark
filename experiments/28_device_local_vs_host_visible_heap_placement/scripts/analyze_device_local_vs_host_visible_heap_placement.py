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
    return latest_compatible_run(candidates)


def _load_frame(path: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(validated_rows(payload, EXPERIMENT_ID, path))
    if frame.empty:
        raise ValueError("Input benchmark JSON has no rows[].")
    frame = frame[frame["experiment_id"] == EXPERIMENT_ID].copy()
    if frame.empty:
        raise ValueError("Input benchmark JSON has no Experiment 28 rows.")

    for column in ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps", "dispatch_count"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
    note_maps = frame["notes"].fillna("").map(_parse_notes)
    frame["placement"] = note_maps.map(lambda notes: notes.get("placement", "unknown"))
    frame["upload_ms"] = pd.to_numeric(note_maps.map(lambda notes: notes.get("upload_ms", "0")), errors="coerce")
    frame["readback_ms"] = pd.to_numeric(note_maps.map(lambda notes: notes.get("readback_ms", "0")), errors="coerce")
    frame["resident_buffer_count"] = pd.to_numeric(
        note_maps.map(lambda notes: notes.get("resident_buffer_count", "0")), errors="coerce"
    )
    frame["chain_kernel_count"] = pd.to_numeric(
        note_maps.map(lambda notes: notes.get("reuse_kernel_count")), errors="coerce"
    ).combine_first(frame["dispatch_count"]).astype("int64")
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
        frame.groupby(["variant", "placement", "problem_size", "dispatch_count", "chain_kernel_count"], as_index=False)
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
    comparison_keys = ["problem_size", "dispatch_count", "chain_kernel_count"]
    baseline = summary[summary["placement"] == "host_visible_direct"][
        ["variant", *comparison_keys, "gpu_ms_median", "end_to_end_ms_median", "gbps_median"]
    ].rename(
        columns={
            "variant": "baseline_variant",
            "gpu_ms_median": "baseline_gpu_ms_median",
            "end_to_end_ms_median": "baseline_end_to_end_ms_median",
            "gbps_median": "baseline_gbps_median",
        }
    )
    relative = summary.merge(baseline, on=comparison_keys, how="left", validate="many_to_one")
    if relative["baseline_variant"].isna().any():
        raise ValueError("Host-visible baseline missing for a problem size and kernel-count case.")
    relative["slowdown_vs_host_visible_gpu_path"] = relative["gpu_ms_median"] / relative["baseline_gpu_ms_median"]
    relative["slowdown_vs_host_visible_harness_wall_clock"] = (
        relative["end_to_end_ms_median"] / relative["baseline_end_to_end_ms_median"]
    )
    relative["gpu_path_gbps_ratio_vs_host_visible"] = relative["gbps_median"] / relative["baseline_gbps_median"]
    return relative[
        [
            "variant",
            "baseline_variant",
            "problem_size",
            "dispatch_count",
            "chain_kernel_count",
            "gpu_ms_median",
            "end_to_end_ms_median",
            "slowdown_vs_host_visible_gpu_path",
            "slowdown_vs_host_visible_harness_wall_clock",
            "gbps_median",
            "gpu_path_gbps_ratio_vs_host_visible",
            "upload_ms_median",
            "readback_ms_median",
        ]
    ]


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "variant",
            "problem_size",
            "dispatch_count",
            "chain_kernel_count",
            "gpu_ms_median",
            "gpu_ms_p95",
            "sample_count",
            "correctness_pass_rate",
        ]
    ].copy()
    stability["p95_to_median_gpu_ms"] = stability["gpu_ms_p95"] / stability["gpu_ms_median"]
    return stability


def _build_reuse_amortization(summary: pd.DataFrame) -> pd.DataFrame:
    """Per kernel-count comparison of per-kernel GPU path cost by payment policy.

    For device_local_staged, upload and readback are paid once per group and the
    K identical kernels amortize them; for host_visible_direct every kernel pays
    full system-memory cost. Per-kernel GPU path time and the staged/direct
    ratio locate the measured reuse crossover.
    """
    reuse = summary[summary["dispatch_count"].astype("Int64") > 1].copy()
    reuse = reuse[reuse["chain_kernel_count"] == reuse["dispatch_count"]]
    per_case = reuse[["variant", "placement", "problem_size", "chain_kernel_count", "gpu_ms_median"]].copy()
    per_case["per_kernel_gpu_path_ms"] = per_case["gpu_ms_median"] / per_case["chain_kernel_count"]
    staged = per_case[per_case["placement"] == "device_local_staged"][
        ["chain_kernel_count", "problem_size", "gpu_ms_median", "per_kernel_gpu_path_ms"]
    ].rename(
        columns={
            "gpu_ms_median": "staged_gpu_path_ms_median",
            "per_kernel_gpu_path_ms": "staged_per_kernel_gpu_path_ms",
        }
    )
    direct = per_case[per_case["placement"] == "host_visible_direct"][
        ["chain_kernel_count", "problem_size", "gpu_ms_median", "per_kernel_gpu_path_ms"]
    ].rename(
        columns={
            "gpu_ms_median": "direct_gpu_path_ms_median",
            "per_kernel_gpu_path_ms": "direct_per_kernel_gpu_path_ms",
        }
    )
    amortization = staged.merge(direct, on=["chain_kernel_count", "problem_size"], how="inner")
    amortization["staged_over_direct_per_kernel_ratio"] = (
        amortization["staged_per_kernel_gpu_path_ms"] / amortization["direct_per_kernel_gpu_path_ms"]
    )
    amortization["staged_amortizes_at_k"] = amortization["staged_over_direct_per_kernel_ratio"] < 1.0
    return amortization[
        [
            "chain_kernel_count",
            "problem_size",
            "staged_gpu_path_ms_median",
            "staged_per_kernel_gpu_path_ms",
            "direct_gpu_path_ms_median",
            "direct_per_kernel_gpu_path_ms",
            "staged_over_direct_per_kernel_ratio",
            "staged_amortizes_at_k",
        ]
    ].sort_values(["problem_size", "chain_kernel_count"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, name: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / name, index=False, float_format="%.6f")
    return TABLES_DIR / name


def main() -> None:
    generated_tables: list[Path] = []
    generated_charts: list[Path | None] = []
    parser = argparse.ArgumentParser(description="Analyze Experiment 28 heap placement data.")
    parser.add_argument("--skip-current", action="store_true")
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current)
    summary = _build_summary(frame)
    relative = _build_relative(summary)
    stability = _build_stability(summary)
    amortization = _build_reuse_amortization(summary)
    generated_tables.append(_write_table(summary, "device_local_vs_host_visible_heap_placement_summary.csv"))
    generated_tables.append(_write_table(relative, "device_local_vs_host_visible_heap_placement_relative.csv"))
    generated_tables.append(_write_table(stability, "device_local_vs_host_visible_heap_placement_stability.csv"))
    generated_tables.append(_write_table(amortization, "device_local_vs_host_visible_reuse_amortization.csv"))
    write_analysis_provenance(TABLES_DIR, [source_path], EXPERIMENT_ID, derived_tables=generated_tables, derived_charts=generated_charts, analysis_tools=[Path(__file__)])
    print(
        f"[ok] Wrote Experiment 28 analysis tables from {source_path.name} on "
        f"{metadata.get('gpu_name', 'unknown_gpu')}."
    )


if __name__ == "__main__":
    main()
