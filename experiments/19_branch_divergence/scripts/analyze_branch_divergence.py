#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "19_branch_divergence"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"

PATTERN_ORDER = {
    "uniform_true": 0,
    "uniform_false": 1,
    "alternating": 2,
    "random_p25": 3,
    "random_p50": 4,
    "random_p75": 5,
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
        raise ValueError("Input benchmark JSON has rows but no Experiment 19 entries.")

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
        ("pattern_mode", "pattern_mode"),
        ("expected_true_probability", "expected_true_probability"),
        ("random_threshold", "random_threshold"),
        ("random_seed", "random_seed"),
        ("branch_work_rounds", "branch_work_rounds"),
        ("true_salt", "true_salt"),
        ("false_salt", "false_salt"),
        ("local_size_x", "local_size_x"),
        ("group_count_x", "group_count_x"),
        ("logical_elements", "logical_elements"),
        ("source_span_bytes", "source_span_bytes"),
        ("destination_span_bytes", "destination_span_bytes"),
        ("bytes_per_element", "bytes_per_element"),
    ]:
        frame[column_name] = note_maps.apply(lambda mapping: mapping.get(note_key))

    frame["pattern_mode"] = frame["pattern_mode"].fillna(frame["variant"]).astype(str)
    frame["pattern_order"] = frame["pattern_mode"].map(lambda value: PATTERN_ORDER.get(str(value), len(PATTERN_ORDER)))
    frame["expected_true_probability"] = pd.to_numeric(frame["expected_true_probability"], errors="coerce")
    frame["random_threshold"] = pd.to_numeric(frame["random_threshold"], errors="coerce")
    frame["random_seed"] = pd.to_numeric(frame["random_seed"], errors="coerce")
    frame["branch_work_rounds"] = pd.to_numeric(frame["branch_work_rounds"], errors="coerce")
    frame["true_salt"] = pd.to_numeric(frame["true_salt"], errors="coerce")
    frame["false_salt"] = pd.to_numeric(frame["false_salt"], errors="coerce")
    frame["local_size_x"] = pd.to_numeric(frame["local_size_x"], errors="coerce")
    frame["group_count_x"] = pd.to_numeric(frame["group_count_x"], errors="coerce")
    frame["logical_elements"] = pd.to_numeric(frame["logical_elements"], errors="coerce")
    frame["source_span_bytes"] = pd.to_numeric(frame["source_span_bytes"], errors="coerce")
    frame["destination_span_bytes"] = pd.to_numeric(frame["destination_span_bytes"], errors="coerce")
    frame["bytes_per_element"] = pd.to_numeric(frame["bytes_per_element"], errors="coerce")

    frame["expected_true_probability"] = frame["expected_true_probability"].fillna(
        frame["pattern_mode"].map(
            {
                "uniform_true": 1.0,
                "uniform_false": 0.0,
                "alternating": 0.5,
                "random_p25": 0.25,
                "random_p50": 0.5,
                "random_p75": 0.75,
            }
        )
    )
    frame["problem_size"] = frame["problem_size"].fillna(0).astype(int)
    frame["iteration"] = frame["iteration"].fillna(0).astype(int)
    frame["dispatch_count"] = frame["dispatch_count"].fillna(1).astype(int)
    frame["random_threshold"] = frame["random_threshold"].fillna(0).astype(int)
    frame["random_seed"] = frame["random_seed"].fillna(0).astype(int)
    frame["branch_work_rounds"] = frame["branch_work_rounds"].fillna(0).astype(int)
    frame["true_salt"] = frame["true_salt"].fillna(0).astype(int)
    frame["false_salt"] = frame["false_salt"].fillna(0).astype(int)
    frame["local_size_x"] = frame["local_size_x"].fillna(256).astype(int)
    frame["group_count_x"] = frame["group_count_x"].fillna(0).astype(int)
    frame["logical_elements"] = frame["logical_elements"].fillna(frame["problem_size"]).astype(int)
    frame["source_span_bytes"] = frame["source_span_bytes"].fillna(frame["logical_elements"] * 4).astype(int)
    frame["destination_span_bytes"] = frame["destination_span_bytes"].fillna(frame["logical_elements"] * 4).astype(int)
    frame["bytes_per_element"] = frame["bytes_per_element"].fillna(8).astype(int)
    frame["pattern_order"] = frame["pattern_order"].astype(int)

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
        raise FileNotFoundError("No benchmark_results.json or archived run JSON found for Experiment 19. Run data collection first.")

    frame, metadata = _load_frame(latest_run)
    return frame, metadata, latest_run


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["variant", "pattern_mode", "pattern_order", "problem_size"], as_index=False)
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
            expected_true_probability=("expected_true_probability", "first"),
            random_threshold=("random_threshold", "first"),
            random_seed=("random_seed", "first"),
            branch_work_rounds=("branch_work_rounds", "first"),
            true_salt=("true_salt", "first"),
            false_salt=("false_salt", "first"),
            local_size_x=("local_size_x", "first"),
            group_count_x=("group_count_x", "first"),
            logical_elements=("logical_elements", "first"),
            source_span_bytes=("source_span_bytes", "first"),
            destination_span_bytes=("destination_span_bytes", "first"),
            bytes_per_element=("bytes_per_element", "first"),
        )
        .sort_values(["pattern_order", "variant", "problem_size"])
        .reset_index(drop=True)
    )

    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    summary["p95_to_median_gpu_ms"] = summary["gpu_ms_p95"] / summary["gpu_ms_median"]
    summary["p95_to_median_gbps"] = summary["gbps_p95"] / summary["gbps_median"]
    summary["branch_entropy_score"] = 1.0 - (summary["expected_true_probability"] - 0.5).abs() * 2.0
    summary["total_payload_bytes"] = summary["logical_elements"] * summary["bytes_per_element"]
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["variant"] == "uniform_true"][
        ["problem_size", "gpu_ms_median", "gbps_median", "throughput_median"]
    ].rename(
        columns={
            "gpu_ms_median": "baseline_gpu_ms_median",
            "gbps_median": "baseline_gbps_median",
            "throughput_median": "baseline_throughput_median",
        }
    )
    relative = summary.merge(baseline, on="problem_size", how="left")
    if relative["baseline_gpu_ms_median"].isna().any():
        raise ValueError("Could not build relative table against uniform_true baseline.")

    relative["slowdown_vs_uniform_true"] = relative["gpu_ms_median"] / relative["baseline_gpu_ms_median"]
    relative["delta_gpu_ms_vs_uniform_true_pct"] = (
        (relative["gpu_ms_median"] - relative["baseline_gpu_ms_median"]) / relative["baseline_gpu_ms_median"]
    ) * 100.0
    relative["gbps_ratio_vs_uniform_true"] = relative["gbps_median"] / relative["baseline_gbps_median"]
    relative["throughput_ratio_vs_uniform_true"] = relative["throughput_median"] / relative["baseline_throughput_median"]

    columns = [
        "variant",
        "pattern_mode",
        "pattern_order",
        "problem_size",
        "gpu_ms_median",
        "slowdown_vs_uniform_true",
        "delta_gpu_ms_vs_uniform_true_pct",
        "gbps_median",
        "gbps_ratio_vs_uniform_true",
        "throughput_median",
        "throughput_ratio_vs_uniform_true",
        "branch_entropy_score",
        "expected_true_probability",
    ]
    return relative[columns].sort_values(["pattern_order", "variant", "problem_size"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "variant",
            "pattern_mode",
            "pattern_order",
            "problem_size",
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
    return stability.sort_values(["pattern_order", "variant", "problem_size"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 19 branch divergence data.")
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
    _write_table(summary, TABLES_DIR / "branch_divergence_summary.csv")
    _write_table(relative, TABLES_DIR / "branch_divergence_relative.csv")
    _write_table(stability, TABLES_DIR / "branch_divergence_stability.csv")

    rows = int(frame.shape[0])
    cases = int(summary.shape[0])
    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(f"[ok] Wrote Experiment 19 analysis tables from {source_label} ({rows} rows, {cases} cases on {gpu_name}).")


if __name__ == "__main__":
    main()
