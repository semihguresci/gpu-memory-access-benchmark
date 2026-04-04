#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_IDS = [
    "01_dispatch_basics",
    "02_local_size_sweep",
    "03_memory_copy_baseline",
    "04_sequential_indexing",
    "05_global_id_mapping_variants",
    "06_aos_vs_soa",
    "07_aosoa_blocked_layout",
    "08_std430_std140_packed",
    "09_vec3_vec4_padding_costs",
    "10_scalar_type_width_sweep",
    "11_coalesced_vs_strided",
    "12_gather_access_pattern",
    "13_scatter_access_pattern",
    "14_read_reuse_cache_locality",
    "15_bandwidth_saturation_sweep",
    "16_shared_memory_tiling",
]


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


def generate_experiment_05(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "05_global_id_mapping_variants"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 05_global_id_mapping_variants"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_global_id_mapping_variants.py")], ROOT)
        run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    else:
        # Regenerate from collected run files only when the latest benchmark JSON is absent.
        run_command(
            [sys.executable, str(scripts_root / "analyze_global_id_mapping_variants.py"), "--skip-current"], ROOT
        )
        print("[info] Skipped plot_results.py because benchmark_results.json is missing.")
    return True


def generate_experiment_06(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "06_aos_vs_soa"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 06_aos_vs_soa"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_aos_vs_soa.py")], ROOT)
        run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    else:
        # Regenerate from collected run files only when the latest benchmark JSON is absent.
        run_command([sys.executable, str(scripts_root / "analyze_aos_vs_soa.py"), "--skip-current"], ROOT)
        print("[info] Skipped plot_results.py because benchmark_results.json is missing.")
    return True


def generate_experiment_07(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "07_aosoa_blocked_layout"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 07_aosoa_blocked_layout"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_aosoa_blocked_layout.py")], ROOT)
        run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    else:
        # Regenerate from collected run files only when the latest benchmark JSON is absent.
        run_command([sys.executable, str(scripts_root / "analyze_aosoa_blocked_layout.py"), "--skip-current"], ROOT)
        print("[info] Skipped plot_results.py because benchmark_results.json is missing.")
    return True


def generate_experiment_08(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "08_std430_std140_packed"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 08_std430_std140_packed"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_std430_std140_packed.py")], ROOT)
        run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    else:
        # Regenerate from collected run files only when the latest benchmark JSON is absent.
        run_command([sys.executable, str(scripts_root / "analyze_std430_std140_packed.py"), "--skip-current"], ROOT)
        print("[info] Skipped plot_results.py because benchmark_results.json is missing.")
    return True


def generate_experiment_09(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "09_vec3_vec4_padding_costs"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 09_vec3_vec4_padding_costs"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_vec3_vec4_padding_costs.py")], ROOT)
    else:
        run_command([sys.executable, str(scripts_root / "analyze_vec3_vec4_padding_costs.py"), "--skip-current"], ROOT)
    return True


def generate_experiment_10(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "10_scalar_type_width_sweep"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 10_scalar_type_width_sweep"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_scalar_type_width_sweep.py")], ROOT)
    else:
        print("[info] Skipped analysis because benchmark_results.json is missing for Experiment 10.")
    return True


def generate_experiment_11(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "11_coalesced_vs_strided"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 11_coalesced_vs_strided"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_coalesced_vs_strided.py")], ROOT)
    else:
        run_command([sys.executable, str(scripts_root / "analyze_coalesced_vs_strided.py"), "--skip-current"], ROOT)

    run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    return True


def generate_experiment_12(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "12_gather_access_pattern"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 12_gather_access_pattern"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_gather_access_pattern.py")], ROOT)
    else:
        run_command([sys.executable, str(scripts_root / "analyze_gather_access_pattern.py"), "--skip-current"], ROOT)

    run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    return True


def generate_experiment_13(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "13_scatter_access_pattern"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 13_scatter_access_pattern"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_scatter_access_pattern.py")], ROOT)
    else:
        run_command([sys.executable, str(scripts_root / "analyze_scatter_access_pattern.py"), "--skip-current"], ROOT)

    run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    return True


def generate_experiment_14(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "14_read_reuse_cache_locality"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 14_read_reuse_cache_locality"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_read_reuse_cache_locality.py")], ROOT)
    else:
        run_command(
            [sys.executable, str(scripts_root / "analyze_read_reuse_cache_locality.py"), "--skip-current"], ROOT
        )

    run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    return True


def generate_experiment_15(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "15_bandwidth_saturation_sweep"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 15_bandwidth_saturation_sweep"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_bandwidth_saturation_sweep.py")], ROOT)
    else:
        run_command([sys.executable, str(scripts_root / "analyze_bandwidth_saturation_sweep.py"), "--skip-current"], ROOT)

    run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    return True


def generate_experiment_16(collect_run: bool) -> bool:
    exp_root = ROOT / "experiments" / "16_shared_memory_tiling"
    scripts_root = exp_root / "scripts"
    benchmark_json = exp_root / "results" / "tables" / "benchmark_results.json"
    runs_root = exp_root / "runs"

    has_runs = runs_root.exists() and any(path.suffix == ".json" for path in runs_root.rglob("*.json"))
    if not benchmark_json.exists() and not has_runs:
        print("[info] No benchmark logs found. Nothing to generate.")
        print(
            "[info] Collect data first with: "
            "python scripts/run_experiment_data_collection.py --experiment 16_shared_memory_tiling"
        )
        return False

    if collect_run and benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "collect_run.py")], ROOT)

    if benchmark_json.exists():
        run_command([sys.executable, str(scripts_root / "analyze_shared_memory_tiling.py")], ROOT)
    else:
        run_command([sys.executable, str(scripts_root / "analyze_shared_memory_tiling.py"), "--skip-current"], ROOT)

    run_command([sys.executable, str(scripts_root / "plot_results.py")], ROOT)
    return True


def generate_experiment(experiment: str, collect_run: bool) -> bool:
    if experiment == "01_dispatch_basics":
        return generate_experiment_01(collect_run=collect_run)
    if experiment == "02_local_size_sweep":
        return generate_experiment_02(collect_run=collect_run)
    if experiment == "03_memory_copy_baseline":
        return generate_experiment_03(collect_run=collect_run)
    if experiment == "04_sequential_indexing":
        return generate_experiment_04(collect_run=collect_run)
    if experiment == "05_global_id_mapping_variants":
        return generate_experiment_05(collect_run=collect_run)
    if experiment == "06_aos_vs_soa":
        return generate_experiment_06(collect_run=collect_run)
    if experiment == "07_aosoa_blocked_layout":
        return generate_experiment_07(collect_run=collect_run)
    if experiment == "08_std430_std140_packed":
        return generate_experiment_08(collect_run=collect_run)
    if experiment == "09_vec3_vec4_padding_costs":
        return generate_experiment_09(collect_run=collect_run)
    if experiment == "10_scalar_type_width_sweep":
        return generate_experiment_10(collect_run=collect_run)
    if experiment == "11_coalesced_vs_strided":
        return generate_experiment_11(collect_run=collect_run)
    if experiment == "12_gather_access_pattern":
        return generate_experiment_12(collect_run=collect_run)
    if experiment == "13_scatter_access_pattern":
        return generate_experiment_13(collect_run=collect_run)
    if experiment == "14_read_reuse_cache_locality":
        return generate_experiment_14(collect_run=collect_run)
    if experiment == "15_bandwidth_saturation_sweep":
        return generate_experiment_15(collect_run=collect_run)
    if experiment == "16_shared_memory_tiling":
        return generate_experiment_16(collect_run=collect_run)
    raise ValueError(f"Unsupported experiment id: {experiment}")


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
        for experiment in selected_experiments:
            generated = generate_experiment(experiment, collect_run=args.collect_run) or generated
    except subprocess.CalledProcessError as exc:
        print(f"[error] Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc

    if generated:
        print("[ok] Artifact generation completed.")


if __name__ == "__main__":
    main()
