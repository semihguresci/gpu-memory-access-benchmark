#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run_command(args: list[str], cwd: Path) -> None:
    printable = " ".join(args)
    print(f"[run] {printable}")
    subprocess.run(args, cwd=str(cwd), check=True)


def generate_experiment_01(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "01_dispatch_basics"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print("[info] Collect data first with: python scripts/run_experiment_data_collection.py")
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_dispatch_basics.py")], ROOT)
        run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    else:
        # Regenerate from collected run files only when the latest benchmark JSON is absent.
        run_command([sys.executable, str(scripts_root / "analyze_dispatch_basics.py"), "--skip-current"], ROOT)
        print("[info] Skipped plot_results.py because benchmark_results.json is missing.")
    return True


def generate_experiment_02(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "02_local_size_sweep"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print("[info] Collect data first with: python scripts/run_experiment_data_collection.py --experiment 02_local_size_sweep")
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_local_size_sweep.py")], ROOT)
        run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    else:
        # Regenerate from collected run files only when the latest benchmark JSON is absent.
        run_command([sys.executable, str(scripts_root / "analyze_local_size_sweep.py"), "--skip-current"], ROOT)
        print("[info] Skipped plot_results.py because benchmark_results.json is missing.")
    return True


def generate_experiment_03(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "03_memory_copy_baseline"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 03_memory_copy_baseline"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_memory_copy_baseline.py")], ROOT)
        run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    else:
        # Regenerate from collected run files only when the latest benchmark JSON is absent.
        run_command([sys.executable, str(scripts_root / "analyze_memory_copy_baseline.py"), "--skip-current"], ROOT)
        print("[info] Skipped plot_results.py because benchmark_results.json is missing.")
    return True


def generate_experiment_04(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "04_sequential_indexing"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 04_sequential_indexing"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_sequential_indexing.py")], ROOT)
        run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    else:
        # Regenerate from collected run files only when the latest benchmark JSON is absent.
        run_command([sys.executable, str(scripts_root / "analyze_sequential_indexing.py"), "--skip-current"], ROOT)
        print("[info] Skipped plot_results.py because benchmark_results.json is missing.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate experiment-local benchmark tables/charts from existing run logs."
    )
    parser.add_argument(
        "--experiment",
        default="01_dispatch_basics",
        choices=["01_dispatch_basics", "02_local_size_sweep", "03_memory_copy_baseline", "04_sequential_indexing"],
        help="Experiment artifact bundle to generate.",
    )
    parser.add_argument(
        "--collect-run",
        action="store_true",
        help="Also collect benchmark_results.json into runs/<device>/<timestamp>.json before analysis (if present).",
    )
    args = parser.parse_args()

    try:
        generated = False
        if args.experiment == "01_dispatch_basics":
            generated = generate_experiment_01(collect_run=args.collect_run)
        elif args.experiment == "02_local_size_sweep":
            generated = generate_experiment_02(collect_run=args.collect_run)
        elif args.experiment == "03_memory_copy_baseline":
            generated = generate_experiment_03(collect_run=args.collect_run)
        elif args.experiment == "04_sequential_indexing":
            generated = generate_experiment_04(collect_run=args.collect_run)
    except subprocess.CalledProcessError as exc:
        print(f"[error] Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc

    if generated:
        print("[ok] Artifact generation completed.")


if __name__ == "__main__":
    main()
