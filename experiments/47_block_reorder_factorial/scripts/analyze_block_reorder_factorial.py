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

EXPERIMENT_ID = "47_block_reorder_factorial"
WIDTH_PATTERN = re.compile(r"^w(\d+)_")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"


def _latest_run_path() -> Path | None:
    candidates = sorted(path for path in RUNS_DIR.rglob("*.json") if path.is_file()) if RUNS_DIR.exists() else []
    return latest_compatible_run(candidates)


def _extract_note_int(notes: str, key: str) -> int | None:
    match = re.search(rf"{key}=(\d+)", notes or "")
    if match is None:
        return None
    return int(match.group(1))


def _extract_checksum(notes: str) -> int | None:
    return _extract_note_int(notes, "result_checksum")


def _variant_block_width(variant: str) -> int | None:
    match = WIDTH_PATTERN.match(variant or "")
    return int(match.group(1)) if match else None


def _parse_residency(notes: str) -> str:
    match = re.search(r"residency=([a-z_]+)", notes or "")
    return match.group(1) if match else "unknown"


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
    frame["residency"] = frame["notes"].astype(str).map(_parse_residency)
    frame["block_width"] = frame["variant"].astype(str).map(_variant_block_width)
    if frame["block_width"].isna().any():
        missing = frame[frame["block_width"].isna()]["variant"].unique().tolist()
        raise ValueError(f"Variant names missing the w<width> prefix: {missing}")
    evicted_rows = frame["residency"].eq("evicted_buffer_sweep")
    if not evicted_rows.any():
        raise ValueError(f"Input benchmark JSON has no evicted_buffer_sweep rows for {EXPERIMENT_ID}.")
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
            residency=("residency", "first"),
            block_width=("block_width", "first"),
            gpu_ms_median=("gpu_ms", "median"),
            gpu_ms_p95=("gpu_ms", _quantile_95),
            throughput_median=("throughput", "median"),
            gbps_median=("gbps", "median"),
        )
        .sort_values(["problem_size", "variant"])
        .reset_index(drop=True)
    )


def _baseline_variant(block_width: int) -> str:
    return f"w{block_width}_blocks_seq_within_ordered"


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    warm = summary[summary["residency"] == "warm"].copy()
    comparison_keys = ["problem_size", "block_width"]
    baseline = warm[warm["variant"] == warm["block_width"].map(lambda width: _baseline_variant(int(width)))].rename(
        columns={
            "variant": "baseline_variant",
            "gpu_ms_median": "baseline_gpu_ms_median",
            "throughput_median": "baseline_throughput_median",
            "gbps_median": "baseline_gbps_median",
        }
    )[
        [
            "baseline_variant",
            "problem_size",
            "block_width",
            "baseline_gpu_ms_median",
            "baseline_throughput_median",
            "baseline_gbps_median",
        ]
    ]
    relative = warm.merge(
        baseline,
        on=comparison_keys,
        how="left",
        validate="many_to_one",
    )
    if relative["baseline_variant"].isna().any():
        raise ValueError("Sequential/ordered baseline missing for a problem size and block width.")
    relative["slowdown_vs_same_width_baseline"] = relative["gpu_ms_median"] / relative["baseline_gpu_ms_median"]
    relative["gbps_ratio_vs_baseline"] = relative["gbps_median"] / relative["baseline_gbps_median"]
    return relative[
        [
            "variant",
            "baseline_variant",
            "problem_size",
            "block_width",
            "residency",
            "gpu_ms_median",
            "slowdown_vs_same_width_baseline",
            "gbps_median",
            "gbps_ratio_vs_baseline",
        ]
    ]


def _build_factor_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """Per (size, width): within-reversal, block-shuffle, and combined time ratios."""
    warm = summary[summary["residency"] == "warm"]
    records: list[dict] = []
    for (problem_size, block_width), group in warm.groupby(["problem_size", "block_width"]):
        medians = dict(zip(group["variant"], group["gpu_ms_median"], strict=True))

        def key(mode: str, mapping: str) -> str:
            return f"w{int(block_width)}_blocks_{mode}_within_{mapping}"

        base = medians.get(key("seq", "ordered"))
        within = medians.get(key("seq", "reversed"))
        shuffle = medians.get(key("shuffled", "ordered"))
        combined = medians.get(key("shuffled", "reversed"))
        if base is None or within is None or shuffle is None or combined is None:
            raise ValueError(f"Factorial cells missing for size={problem_size}, width={block_width}.")
        records.append(
            {
                "problem_size": problem_size,
                "block_width": int(block_width),
                "within_reversal_ratio": within / base,
                "block_shuffle_ratio": shuffle / base,
                "combined_ratio": combined / base,
                "additivity_ratio": (combined / base) / ((within / base) * (shuffle / base)),
            }
        )
    return pd.DataFrame(records).sort_values(["problem_size", "block_width"]).reset_index(drop=True)


def _build_residency_effects(summary: pd.DataFrame) -> pd.DataFrame:
    warm = summary[summary["residency"] == "warm"][
        ["variant", "problem_size", "gpu_ms_median"]
    ].rename(columns={"variant": "warm_variant", "gpu_ms_median": "warm_gpu_ms_median"})
    evicted = summary[summary["residency"] == "evicted_buffer_sweep"][
        ["variant", "problem_size", "gpu_ms_median", "block_width"]
    ].rename(columns={"variant": "evicted_variant", "gpu_ms_median": "evicted_gpu_ms_median"})
    evicted["base_variant"] = evicted["evicted_variant"].str.replace("_evicted$", "", regex=True)
    merged = evicted.merge(
        warm, left_on=["base_variant", "problem_size"], right_on=["warm_variant", "problem_size"], how="inner"
    )
    merged["evicted_over_warm_ratio"] = merged["evicted_gpu_ms_median"] / merged["warm_gpu_ms_median"]
    return merged[
        [
            "base_variant",
            "evicted_variant",
            "problem_size",
            "block_width",
            "warm_gpu_ms_median",
            "evicted_gpu_ms_median",
            "evicted_over_warm_ratio",
        ]
    ]


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
    stability = summary[["variant", "problem_size", "block_width", "residency", "gpu_ms_median", "gpu_ms_p95"]].copy()
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
    effects = _build_factor_effects(summary)
    residency = _build_residency_effects(summary)
    checksums = _build_checksums(frame)
    stability = _build_stability(summary)
    generated_tables.append(_write_table(summary, "block_reorder_factorial_summary.csv"))
    generated_tables.append(_write_table(relative, "block_reorder_factorial_relative.csv"))
    generated_tables.append(_write_table(effects, "block_reorder_factorial_effects.csv"))
    generated_tables.append(_write_table(residency, "block_reorder_factorial_residency.csv"))
    generated_tables.append(_write_table(checksums, "block_reorder_factorial_checksums.csv"))
    generated_tables.append(_write_table(stability, "block_reorder_factorial_stability.csv"))
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
