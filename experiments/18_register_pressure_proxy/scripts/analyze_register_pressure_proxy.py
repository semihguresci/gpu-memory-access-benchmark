#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_ID = "18_register_pressure_proxy"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"


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
        raise ValueError("Input benchmark JSON has rows but no Experiment 18 entries.")

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
        ("temporary_count", "temporary_count"),
        ("round_count", "round_count"),
        ("local_size_x", "local_size_x"),
        ("group_count_x", "group_count_x"),
        ("logical_elements", "logical_elements"),
        ("source_span_bytes", "source_span_bytes"),
        ("destination_span_bytes", "destination_span_bytes"),
        ("payload_bytes_per_element", "payload_bytes_per_element"),
    ]:
        frame[column_name] = pd.to_numeric(note_maps.apply(lambda mapping: mapping.get(note_key)), errors="coerce")

    frame["temporary_count"] = frame["temporary_count"].fillna(frame["variant"].str.extract(r"temp_(\d+)")[0])
    frame["temporary_count"] = pd.to_numeric(frame["temporary_count"], errors="coerce")
    if frame["temporary_count"].isna().any():
        raise ValueError("Could not determine temporary_count for one or more rows.")
    frame["temporary_count"] = frame["temporary_count"].astype(int)

    frame["round_count"] = frame["round_count"].fillna(8)
    frame["round_count"] = pd.to_numeric(frame["round_count"], errors="coerce")
    if frame["round_count"].isna().any():
        raise ValueError("Could not determine round_count for one or more rows.")
    frame["round_count"] = frame["round_count"].astype(int)

    frame["local_size_x"] = frame["local_size_x"].fillna(256)
    frame["local_size_x"] = pd.to_numeric(frame["local_size_x"], errors="coerce")
    if frame["local_size_x"].isna().any():
        raise ValueError("Could not determine local_size_x for one or more rows.")
    frame["local_size_x"] = frame["local_size_x"].astype(int)

    frame["group_count_x"] = frame["group_count_x"].fillna(0)
    frame["group_count_x"] = pd.to_numeric(frame["group_count_x"], errors="coerce")
    if frame["group_count_x"].isna().any():
        raise ValueError("Could not determine group_count_x for one or more rows.")
    frame["group_count_x"] = frame["group_count_x"].astype(int)

    frame["logical_elements"] = frame["logical_elements"].fillna(frame["problem_size"])
    frame["logical_elements"] = pd.to_numeric(frame["logical_elements"], errors="coerce")
    if frame["logical_elements"].isna().any():
        raise ValueError("Could not determine logical_elements for one or more rows.")
    frame["logical_elements"] = frame["logical_elements"].astype(int)

    frame["source_span_bytes"] = frame["source_span_bytes"].fillna(frame["logical_elements"] * 4)
    frame["source_span_bytes"] = pd.to_numeric(frame["source_span_bytes"], errors="coerce")
    if frame["source_span_bytes"].isna().any():
        raise ValueError("Could not determine source_span_bytes for one or more rows.")
    frame["source_span_bytes"] = frame["source_span_bytes"].astype(int)

    frame["destination_span_bytes"] = frame["destination_span_bytes"].fillna(frame["source_span_bytes"])
    frame["destination_span_bytes"] = pd.to_numeric(frame["destination_span_bytes"], errors="coerce")
    if frame["destination_span_bytes"].isna().any():
        raise ValueError("Could not determine destination_span_bytes for one or more rows.")
    frame["destination_span_bytes"] = frame["destination_span_bytes"].astype(int)

    frame["payload_bytes_per_element"] = frame["payload_bytes_per_element"].fillna(8)
    frame["payload_bytes_per_element"] = pd.to_numeric(frame["payload_bytes_per_element"], errors="coerce")
    if frame["payload_bytes_per_element"].isna().any():
        raise ValueError("Could not determine payload_bytes_per_element for one or more rows.")
    frame["payload_bytes_per_element"] = frame["payload_bytes_per_element"].astype(int)

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
            "No benchmark_results.json or archived run JSON found for Experiment 18. Run data collection first."
        )

    frame, metadata = _load_frame(latest_run)
    return frame, metadata, latest_run


def _quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def _build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["variant", "temporary_count"], as_index=False)
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
            problem_size=("problem_size", "first"),
            round_count=("round_count", "first"),
            local_size_x=("local_size_x", "first"),
            group_count_x=("group_count_x", "first"),
            source_span_bytes=("source_span_bytes", "first"),
            destination_span_bytes=("destination_span_bytes", "first"),
            payload_bytes_per_element=("payload_bytes_per_element", "first"),
        )
        .sort_values(["temporary_count", "variant"])
        .reset_index(drop=True)
    )

    summary["gpu_ms_cv"] = summary["gpu_ms_std"] / summary["gpu_ms_mean"]
    summary["gbps_cv"] = summary["gbps_std"] / summary["gbps_mean"]
    summary["p95_to_median_gpu_ms"] = summary["gpu_ms_p95"] / summary["gpu_ms_median"]
    summary["p95_to_median_gbps"] = summary["gbps_p95"] / summary["gbps_median"]
    summary["payload_bytes_total"] = summary["problem_size"] * summary["payload_bytes_per_element"]
    return summary


def _build_relative(summary: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = summary[summary["temporary_count"] == summary["temporary_count"].min()]
    if baseline_rows.empty:
        raise ValueError("Could not determine baseline temporary_count for relative table.")

    baseline = baseline_rows.iloc[0]
    relative = summary.copy()
    relative["speedup_vs_baseline"] = float(baseline["gpu_ms_median"]) / relative["gpu_ms_median"]
    relative["gpu_delta_pct_vs_baseline"] = (
        (relative["gpu_ms_median"] - float(baseline["gpu_ms_median"])) / float(baseline["gpu_ms_median"])
    ) * 100.0
    relative["gbps_ratio_vs_baseline"] = relative["gbps_median"] / float(baseline["gbps_median"])
    relative["throughput_ratio_vs_baseline"] = relative["throughput_median"] / float(baseline["throughput_median"])

    columns = [
        "variant",
        "temporary_count",
        "problem_size",
        "gpu_ms_median",
        "speedup_vs_baseline",
        "gpu_delta_pct_vs_baseline",
        "gbps_median",
        "gbps_ratio_vs_baseline",
        "throughput_median",
        "throughput_ratio_vs_baseline",
    ]
    return relative[columns].sort_values(["temporary_count", "variant"]).reset_index(drop=True)


def _build_stability(summary: pd.DataFrame) -> pd.DataFrame:
    stability = summary[
        [
            "variant",
            "temporary_count",
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
    return stability.sort_values(["temporary_count", "variant"]).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 18 register pressure proxy data.")
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
    _write_table(summary, TABLES_DIR / "register_pressure_proxy_summary.csv")
    _write_table(relative, TABLES_DIR / "register_pressure_proxy_relative.csv")
    _write_table(stability, TABLES_DIR / "register_pressure_proxy_stability.csv")

    rows = int(frame.shape[0])
    variants = int(summary.shape[0])
    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(f"[ok] Wrote Experiment 18 analysis tables from {source_label} ({rows} rows, {variants} cases on {gpu_name}).")


if __name__ == "__main__":
    main()
