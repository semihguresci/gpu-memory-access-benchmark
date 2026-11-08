#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from experiment_manifest import ROOT
from experiment_workflow import enabled_experiment_ids, load_enabled_experiment_configs, resolve_binary


EXPERIMENT_CONFIG = load_enabled_experiment_configs()
EXPERIMENT_IDS = enabled_experiment_ids()

# Convenience scratch-size presets for larger working sets. Values are forwarded
# verbatim as --size arguments, so every experiment keeps its own budget split.
SIZE_PRESETS: dict[str, str] = {
    "tiny": "4M",
    "small": "32M",
    "medium": "128M",
    "large": "256M",
    "xlarge": "1G",
}


def resolve_scratch_size(size: str | None, size_preset: str | None) -> str | None:
    """Resolve CLI scratch-size arguments to one --size value (or None for default)."""
    if size is not None and size_preset is not None:
        raise ValueError("--size and --size-preset cannot be combined.")
    if size_preset is not None:
        try:
            return SIZE_PRESETS[size_preset]
        except KeyError as exc:
            available = ", ".join(sorted(SIZE_PRESETS))
            raise ValueError(f"Unknown --size-preset '{size_preset}'. Available presets: {available}") from exc
    return size


def build_run_label(label: str, run_index: int, run_count: int) -> str:
    """Label collected runs so independent repetitions archive under distinct files."""
    if run_count <= 1:
        return label
    suffix = f"run{run_index}"
    return f"{label}_{suffix}" if label else suffix


def _run_command(args: list[str]) -> None:
    print(f"[run] {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=str(ROOT), check=True)


def _hash_files(paths: list[Path], relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(relative_to).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_provenance(binary: Path) -> dict:
    source_files = [ROOT / "CMakeLists.txt", ROOT / "CMakePresets.json"]
    for directory in ("include", "src", "shaders", "cmake", "config", "scripts"):
        source_files.extend(
            path for path in (ROOT / directory).rglob("*")
            if path.is_file() and path.suffix in {".cpp", ".hpp", ".comp", ".glsl", ".cmake", ".in", ".json", ".py"}
        )
    source_files.extend((ROOT / "experiments").glob("*/scripts/*.py"))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, capture_output=True)
    return {
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_tree_sha256": _hash_files(source_files, ROOT),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "working_tree_dirty": dirty.returncode != 0,
    }


def _attach_provenance(output_path: Path, provenance: dict) -> None:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    metadata = payload.setdefault("metadata", {})
    record = dict(provenance)
    shader_directory = Path(metadata.get("shader_binary_directory", ""))
    if metadata.get("shader_binary_directory") and shader_directory.is_dir():
        shader_files = list(shader_directory.rglob("*.spv"))
        record["shader_bundle_sha256"] = _hash_files(shader_files, shader_directory)
        record["shader_bundle_file_count"] = len(shader_files)
    metadata["provenance"] = record
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_one_collection(binary: Path, provenance: dict, experiment_id: str, benchmark_args: list[str], run_size: str,
                        collect: bool, label: str) -> None:
    config = EXPERIMENT_CONFIG[experiment_id]
    output_path = config.output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] Experiment {experiment_id} (size={run_size})", flush=True)

    benchmark_cmd = [
        str(binary),
        "--experiment",
        experiment_id,
        *benchmark_args,
        "--output",
        str(output_path),
    ]

    _run_command(benchmark_cmd)
    _attach_provenance(output_path, provenance)

    if collect:
        collect_cmd = [sys.executable, str(config.collect_script), "--input", str(output_path)]
        if label:
            collect_cmd.extend(["--label", label])
        _run_command(collect_cmd)


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
    size_arguments = parser.add_mutually_exclusive_group()
    size_arguments.add_argument(
        "--size",
        type=str,
        default=None,
        help="Scratch buffer size for benchmark run (e.g. 4M). If omitted, uses experiment default.",
    )
    size_arguments.add_argument(
        "--size-preset",
        type=str,
        default=None,
        choices=sorted(SIZE_PRESETS),
        help="Named scratch-size preset; expanded to the mapped --size value. "
        f"Available: {', '.join(f'{name}={value}' for name, value in sorted(SIZE_PRESETS.items()))}.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent benchmark invocations to run and archive (>= 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Physical device selection forwarded to the benchmark (index or name substring).",
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

    if args.runs < 1:
        parser.error("--runs must be at least 1.")

    try:
        selected_size = resolve_scratch_size(args.size, args.size_preset)
    except ValueError as exc:
        parser.error(str(exc))

    binary = resolve_binary(args.binary)
    provenance = _source_provenance(binary)
    selected_experiment_ids = list(EXPERIMENT_IDS) if args.experiment == "all" else [args.experiment]
    print(f"[info] Running data collection for {len(selected_experiment_ids)} experiment(s).", flush=True)

    for experiment_id in selected_experiment_ids:
        default_size = EXPERIMENT_CONFIG[experiment_id].default_size
        run_size = selected_size if selected_size is not None else default_size
        for run_index in range(1, args.runs + 1):
            print(
                f"[info] Run {run_index}/{args.runs} for {experiment_id} (size={run_size}).", flush=True
            )

            benchmark_args = [
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
                "--size",
                run_size,
            ]
            if args.device is not None:
                benchmark_args.extend(["--device", args.device])
            if args.validation:
                benchmark_args.append("--validation")
            if args.verbose_progress:
                benchmark_args.append("--verbose-progress")

            _run_one_collection(
                binary=binary,
                provenance=provenance,
                experiment_id=experiment_id,
                benchmark_args=benchmark_args,
                run_size=run_size,
                collect=not args.no_collect,
                label=build_run_label(args.label, run_index, args.runs),
            )

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
