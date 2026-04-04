#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = ROOT / "results" / "tables" / "benchmark_results.json"
DEFAULT_RUNS_DIR = ROOT / "runs"
EXPERIMENT_ID = "15_bandwidth_saturation_sweep"


def _slugify(value: str, fallback: str = "unknown") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or fallback


def _parse_iso8601(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _build_output_path(payload: dict, runs_dir: Path, label: str) -> Path:
    metadata = payload.get("metadata", {})
    gpu_name = str(metadata.get("gpu_name", "unknown_gpu"))
    gpu_slug = _slugify(gpu_name, fallback="unknown_gpu")

    exported_at_utc = str(metadata.get("exported_at_utc", ""))
    timestamp = _parse_iso8601(exported_at_utc) or datetime.now(timezone.utc)
    timestamp_token = timestamp.strftime("%Y%m%d_%H%M%SZ")

    label_slug = _slugify(label) if label else ""
    filename = f"{timestamp_token}.json" if not label_slug else f"{timestamp_token}_{label_slug}.json"

    output_dir = runs_dir / gpu_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / filename
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        alternate_name = (
            f"{timestamp_token}_{suffix}.json" if not label_slug else f"{timestamp_token}_{label_slug}_{suffix}.json"
        )
        alternate = output_dir / alternate_name
        if not alternate.exists():
            return alternate
        suffix += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Experiment 15 JSON output into runs/<device>/...")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input benchmark JSON path (default: experiments/15_bandwidth_saturation_sweep/results/tables/benchmark_results.json).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Destination runs directory (default: experiments/15_bandwidth_saturation_sweep/runs).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional run label appended to the filename.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input JSON has no rows[] payload.")

    if not any(str(row.get("experiment_id", "")) == EXPERIMENT_ID for row in rows):
        raise ValueError(
            "Input JSON has rows but no Experiment 15 entries. Run with --experiment 15_bandwidth_saturation_sweep first."
        )

    output_path = _build_output_path(payload, args.runs_dir, args.label)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    run_id = output_path.resolve().relative_to(ROOT.resolve()).as_posix()
    print(f"Collected run: {run_id}")


if __name__ == "__main__":
    main()
