#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys

from experiment_manifest import ROOT, load_experiment_manifest


EXPERIMENT_CONFIG = {
    str(experiment["id"]): experiment for experiment in load_experiment_manifest() if bool(experiment["enabled"])
}
EXPERIMENT_IDS = list(EXPERIMENT_CONFIG)


def run_command(args: list[str]) -> None:
    printable = " ".join(args)
    print(f"[run] {printable}")
    subprocess.run(args, cwd=str(ROOT), check=True)


def generate_experiment(experiment_id: str, collect_run: bool) -> bool:
    config = EXPERIMENT_CONFIG[experiment_id]
    exp_root = ROOT / "experiments" / experiment_id
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            f"python scripts/run_experiment_data_collection.py --experiment {experiment_id}"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")])

    analysis_script = scripts_root / str(config["analysis_script"])
    if benchmark_json.exists():
        run_command([sys.executable, str(analysis_script)])
    else:
        # Some analyzers can rebuild derived tables from archived runs even when the
        # current benchmark export has been cleaned from results/tables.
        if bool(config["supports_skip_current"]):
            run_command([sys.executable, str(analysis_script), "--skip-current"])
        else:
            print(f"[info] Skipped analysis because benchmark_results.json is missing for {experiment_id}.")

    if bool(config["has_plot_script"]):
        # A few experiments intentionally plot from aggregated run outputs after
        # --skip-current analysis, so plotting cannot depend only on a fresh export.
        if benchmark_json.exists() or bool(config["plot_after_skip_current"]):
            run_command([sys.executable, str(scripts_root / "plot_results.py")])
        else:
            print("[info] Skipped plot_results.py because benchmark_results.json is missing.")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate experiment-local benchmark tables/charts from existing run logs."
    )
    parser.add_argument(
        "--experiment",
        default="01_dispatch_basics",
        choices=["all", *EXPERIMENT_IDS],
        help="Experiment artifact bundle to generate.",
    )
    parser.add_argument(
        "--collect-run",
        action="store_true",
        help="Also collect benchmark_results.json into runs/<device>/<timestamp>.json before analysis (if present).",
    )
    args = parser.parse_args()

    try:
        selected_experiments = EXPERIMENT_IDS if args.experiment == "all" else [args.experiment]
        generated = False
        for experiment_id in selected_experiments:
            generated = generate_experiment(experiment_id, collect_run=args.collect_run) or generated
    except subprocess.CalledProcessError as exc:
        print(f"[error] Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc

    if generated:
        print("[ok] Artifact generation completed.")


if __name__ == "__main__":
    main()
