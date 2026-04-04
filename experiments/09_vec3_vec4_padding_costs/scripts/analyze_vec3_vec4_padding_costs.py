#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENT_ID = "09_vec3_vec4_padding_costs"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNS_DIR = ROOT / "runs"
TABLES_DIR = ROOT / "results" / "tables"
CHARTS_DIR = ROOT / "results" / "charts"
CURRENT_JSON = TABLES_DIR / "benchmark_results.json"

VARIANT_ORDER = {
    "split_scalars": 0,
    "vec4": 1,
    "vec3_padded": 2,
}

VARIANT_LABELS = {
    "split_scalars": "split_scalars",
    "vec4": "vec4",
    "vec3_padded": "vec3_padded",
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
        raise ValueError("Input benchmark JSON has rows but no Experiment 09 entries.")

    numeric_columns = ["problem_size", "iteration", "gpu_ms", "end_to_end_ms", "throughput", "gbps"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

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
        ("storage_bytes_per_particle", "storage_bytes_per_particle"),
        ("logical_bytes_per_particle", "logical_bytes_per_particle"),
        ("alignment_waste_ratio", "alignment_waste_ratio"),
    ]:
        frame[column_name] = pd.to_numeric(note_maps.apply(lambda mapping: mapping.get(note_key)), errors="coerce")

    frame["storage_bytes_per_particle"] = frame["storage_bytes_per_particle"].fillna(0).astype(int)
    frame["logical_bytes_per_particle"] = frame["logical_bytes_per_particle"].fillna(44).astype(int)
    frame["alignment_waste_ratio"] = frame["alignment_waste_ratio"].fillna(0.0).astype(float)
    frame["variant_order"] = frame["variant"].map(lambda variant: VARIANT_ORDER.get(variant, len(VARIANT_ORDER)))

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
            "No benchmark_results.json or archived run JSON found for Experiment 09. "
            "Run data collection first."
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
            gbps_median=("gbps", "median"),
            throughput_median=("throughput", "median"),
            storage_bytes_per_particle=("storage_bytes_per_particle", "first"),
            logical_bytes_per_particle=("logical_bytes_per_particle", "first"),
            alignment_waste_ratio=("alignment_waste_ratio", "first"),
            variant_order=("variant_order", "first"),
        )
        .sort_values(["problem_size", "variant_order", "variant"])
        .reset_index(drop=True)
    )
    return summary


def _write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.6f")


def _plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_frame = summary.sort_values(["variant_order", "problem_size", "variant"])
    for variant, variant_frame in plot_frame.groupby("variant"):
        ordered = variant_frame.sort_values("problem_size")
        ax.plot(
            ordered["problem_size"].astype(int).to_list(),
            ordered["gpu_ms_median"].astype(float).to_list(),
            marker="o",
            linewidth=2.0,
            label=VARIANT_LABELS.get(str(variant), str(variant)),
        )

    ax.set_title("Experiment 09: Median GPU Time by Problem Size")
    ax.set_xlabel("problem_size")
    ax.set_ylabel("GPU ms (median)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Experiment 09 vec3 vs vec4 padding cost data.")
    parser.add_argument(
        "--skip-current",
        action="store_true",
        help="Prefer the latest archived run under runs/ instead of results/tables/benchmark_results.json.",
    )
    args = parser.parse_args()

    frame, metadata, source_path = _load_source_frame(skip_current=args.skip_current)
    summary = _build_summary(frame)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    _write_table(summary, TABLES_DIR / "vec3_vec4_padding_summary.csv")
    _plot_summary(summary, CHARTS_DIR / "vec3_vec4_padding_median_gpu_ms.svg")

    rows = int(frame.shape[0])
    variants = int(summary["variant"].nunique())
    source_label = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    print(
        f"[ok] Wrote Experiment 09 summary artifacts from {source_label} "
        f"({rows} rows, {variants} variants on {gpu_name})."
    )


if __name__ == "__main__":
    main()
