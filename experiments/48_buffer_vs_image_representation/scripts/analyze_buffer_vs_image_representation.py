#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
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

EXPERIMENT_ID = "48_buffer_vs_image_representation"
BASELINE_VARIANT = "row_major_buffer_neighborhood_clamped"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"


def _latest_run_path() -> Path | None:
    candidates = sorted(path for path in RUNS_DIR.rglob("*.json") if path.is_file()) if RUNS_DIR.exists() else []
    return latest_compatible_run(candidates)


def _extract_note(notes: str, key: str) -> str | None:
    match = re.search(rf"{key}=([^;]+)", notes or "")
    return match.group(1) if match else None


def _extract_note_int(notes: str, key: str) -> int | None:
    text = _extract_note(notes, key)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _extract_checksum(notes: str) -> int | None:
    return _extract_note_int(notes, "result_checksum")


def _load_frame(path: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(validated_rows(payload, EXPERIMENT_ID, path))
    if frame.empty:
        raise ValueError("Input benchmark JSON has no rows[].")
    frame = frame[frame["experiment_id"] == EXPERIMENT_ID].copy()
    if frame.empty:
        raise ValueError(f"Input benchmark JSON has no {EXPERIMENT_ID} rows.")
    for column in ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["correctness_pass"] = frame["correctness_pass"].astype(str).str.lower().isin(["1", "true", "yes"])
    frame["result_checksum"] = frame["notes"].astype(str).map(_extract_checksum)
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
    baseline = summary[summary["variant"] == BASELINE_VARIANT][
        ["problem_size", "gpu_ms_median", "throughput_median", "gbps_median"]
    ].rename(
        columns={
            "gpu_ms_median": "baseline_gpu_ms_median",
            "throughput_median": "baseline_throughput_median",
            "gbps_median": "baseline_gbps_median",
        }
    )
    relative = summary.merge(baseline, on="problem_size", how="inner")
    relative["slowdown_vs_row_buffer_neighborhood"] = relative["gpu_ms_median"] / relative["baseline_gpu_ms_median"]
    relative["gbps_ratio_vs_row_buffer_neighborhood"] = relative["gbps_median"] / relative["baseline_gbps_median"]
    return relative[
        [
            "variant",
            "problem_size",
            "gpu_ms_median",
            "slowdown_vs_row_buffer_neighborhood",
            "throughput_median",
            "gbps_median",
            "gbps_ratio_vs_row_buffer_neighborhood",
        ]
    ]


def _split_representation(variant: str) -> tuple[str, str]:
    for representation in ("row_major_buffer", "column_major_buffer", "storage_image", "sampled_image"):
        if variant.startswith(representation + "_"):
            mode = variant[len(representation) + 1:]
            return representation, mode
    raise ValueError(f"Variant name does not parse into representation and mode: {variant}")


def _build_representation_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """Per (size, mode): each representation's median time ratio vs the row buffer."""
    rows: list[dict] = []
    for (problem_size, variant), group in summary.groupby(["problem_size", "variant"]):
        representation, mode = _split_representation(str(variant))
        rows.append(
            {
                "problem_size": int(problem_size),
                "representation": representation,
                "access_mode": mode,
                "gpu_ms_median": float(group["gpu_ms_median"].iloc[0]),
                "gbps_median": float(group["gbps_median"].iloc[0]),
            }
        )
    flat = pd.DataFrame(rows)
    baseline = flat[flat["representation"] == "row_major_buffer"][
        ["problem_size", "access_mode", "gpu_ms_median", "gbps_median"]
    ].rename(columns={"gpu_ms_median": "baseline_gpu_ms_median", "gbps_median": "baseline_gbps_median"})
    merged = flat.merge(baseline, on=["problem_size", "access_mode"], how="inner")
    merged["gpu_time_ratio_vs_row_buffer"] = merged["gpu_ms_median"] / merged["baseline_gpu_ms_median"]
    merged["gbps_ratio_vs_row_buffer"] = merged["gbps_median"] / merged["baseline_gbps_median"]
    return merged[
        [
            "problem_size",
            "representation",
            "access_mode",
            "gpu_ms_median",
            "gpu_time_ratio_vs_row_buffer",
            "gbps_median",
            "gbps_ratio_vs_row_buffer",
        ]
    ].sort_values(["problem_size", "access_mode", "representation"]).reset_index(drop=True)


def _build_checksums(frame: pd.DataFrame) -> pd.DataFrame:
    checksums = (
        frame.groupby(["variant", "problem_size"], as_index=False)
        .agg(
            checksum_unique=("result_checksum", "nunique"),
            first_checksum=("result_checksum", "first"),
        )
        .sort_values(["problem_size", "variant"])
        .reset_index(drop=True)
    )
    checksums["checksum_stable"] = checksums["checksum_unique"] == 1
    return checksums


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[["variant", "problem_size", "gpu_ms_median", "gpu_ms_p95"]].copy()
    stability["p95_to_median_gpu_ms"] = stability["gpu_ms_p95"] / stability["gpu_ms_median"]
    return stability


def _write_table(frame: pd.DataFrame, name: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES_DIR / name, index=False, float_format="%.6f")
    return TABLES_DIR / name


def main() -> None:
    generated_tables: list[Path] = []
    generated_charts: list[Path | None] = []
    parser = argparse.ArgumentParser(description=f"Analyze {EXPERIMENT_ID} data.")
    parser.add_argument("--skip-current", action="store_true")
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current)
    summary = _build_summary(frame)
    relative = _build_relative(summary)
    effects = _build_representation_effects(summary)
    checksums = _build_checksums(frame)
    stability = _build_stability(summary)
    generated_tables.append(_write_table(summary, "buffer_vs_image_representation_summary.csv"))
    generated_tables.append(_write_table(relative, "buffer_vs_image_representation_relative.csv"))
    generated_tables.append(_write_table(effects, "buffer_vs_image_representation_representation_effects.csv"))
    generated_tables.append(_write_table(checksums, "buffer_vs_image_representation_checksums.csv"))
    generated_tables.append(_write_table(stability, "buffer_vs_image_representation_stability.csv"))
    write_analysis_provenance(
        TABLES_DIR, [source_path], EXPERIMENT_ID,
        derived_tables=generated_tables, derived_charts=generated_charts, analysis_tools=[Path(__file__)],
    )
    print(
        f"[ok] Wrote {EXPERIMENT_ID} analysis tables from {source_path.name} on "
        f"{metadata.get('gpu_name', 'unknown_gpu')}."
    )


if __name__ == "__main__":
    main()
