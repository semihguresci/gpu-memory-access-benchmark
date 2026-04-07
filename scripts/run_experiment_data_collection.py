#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiment_manifest import ROOT, load_experiment_manifest


EXPERIMENT_CONFIG = {
    str(experiment["id"]): {
        "output": ROOT / "experiments" / str(experiment["id"]) / "results" / "tables" / "benchmark_results.json",
        "collect_script": ROOT / "experiments" / str(experiment["id"]) / "scripts" / "collect_run.py",
        "default_size": str(experiment["default_size"]),
    }
    for experiment in load_experiment_manifest()
    if bool(experiment["enabled"])
}
EXPERIMENT_IDS = tuple(EXPERIMENT_CONFIG)


def _resolve_binary(explicit_path: str | None) -> Path:
    if explicit_path:
        binary = Path(explicit_path)
        if not binary.is_absolute():
            binary = (ROOT / binary).resolve()
        if not binary.exists():
            raise FileNotFoundError(f"Benchmark binary not found: {binary}")
        return binary

    candidates = [
        ROOT / "build-tests-vs" / "Release" / "gpu_memory_layout_experiments.exe",
        ROOT / "build-tests-vs" / "Debug" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "Release" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "Debug" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "windows-x64" / "Release" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "windows-x64" / "Debug" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "gpu_memory_layout_experiments",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find benchmark binary. Build first, for example:\n"
        "  cmake --build build --config Release"
    )


def _run_command(args: list[str]) -> None:
    print(f"[run] {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark and collect fresh run data (no chart generation).")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", *EXPERIMENT_IDS],
        help="Experiment id to run and collect, or 'all' to run every experiment.",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help="Path to benchmark executable. Defaults to build/Release if available.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Timed iterations for benchmark run.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations for benchmark run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="Scratch buffer size for benchmark run (e.g. 4M). If omitted, uses experiment default.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Enable Vulkan validation layers during benchmark run.",
    )
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        help="Enable verbose per-stage benchmark progress logs.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label suffix for collected run filename.",
    )
    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="Skip collecting the run into experiments/<experiment>/runs.",
    )
    args = parser.parse_args()

    binary = _resolve_binary(args.binary)
    selected_experiment_ids = list(EXPERIMENT_IDS) if args.experiment == "all" else [args.experiment]
    print(f"[info] Running data collection for {len(selected_experiment_ids)} experiment(s).", flush=True)

    for experiment_id in selected_experiment_ids:
        config = EXPERIMENT_CONFIG[experiment_id]
        output_path: Path = config["output"]
        collect_script: Path = config["collect_script"]
        selected_size: str = args.size if args.size is not None else str(config["default_size"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[info] Experiment {experiment_id} (size={selected_size})", flush=True)

        benchmark_cmd = [
            str(binary),
            "--experiment",
            experiment_id,
            "--iterations",
            str(args.iterations),
            "--warmup",
            str(args.warmup),
            "--size",
            selected_size,
            "--output",
            str(output_path),
        ]
        if args.validation:
            benchmark_cmd.append("--validation")
        if args.verbose_progress:
            benchmark_cmd.append("--verbose-progress")

        _run_command(benchmark_cmd)

        if not args.no_collect:
            collect_cmd = [sys.executable, str(collect_script), "--input", str(output_path)]
            if args.label:
                collect_cmd.extend(["--label", args.label])
            _run_command(collect_cmd)

    print("[ok] Data collection completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except subprocess.CalledProcessError as exc:
        print(f"[error] Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
