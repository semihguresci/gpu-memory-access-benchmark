#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiment_manifest import ROOT, load_experiment_manifest


@dataclass(frozen=True, slots=True)
class ExperimentWorkflowConfig:
    id: str
    display_name: str
    category: str
    docs_group: str
    default_size: str
    output_path: Path
    experiment_root: Path
    scripts_root: Path
    collect_script: Path
    analysis_script: Path
    plot_script: Path
    has_plot_script: bool
    supports_skip_current: bool
    plot_after_skip_current: bool
    plan_title: str
    plan_description: str


def load_enabled_experiment_configs() -> dict[str, ExperimentWorkflowConfig]:
    # Python tooling should derive its runnable experiment set from the shared
    # manifest so build, CLI, and GUI registration stay aligned.
    configs: dict[str, ExperimentWorkflowConfig] = {}
    for experiment in load_experiment_manifest():
        if not bool(experiment["enabled"]):
            continue

        experiment_id = str(experiment["id"])
        experiment_root = ROOT / "experiments" / experiment_id
        scripts_root = experiment_root / "scripts"
        configs[experiment_id] = ExperimentWorkflowConfig(
            id=experiment_id,
            display_name=str(experiment["display_name"]),
            category=str(experiment["category"]),
            docs_group=str(experiment["docs_group"]),
            default_size=str(experiment["default_size"]),
            output_path=experiment_root / "results" / "tables" / "benchmark_results.json",
            experiment_root=experiment_root,
            scripts_root=scripts_root,
            collect_script=scripts_root / "collect_run.py",
            analysis_script=scripts_root / str(experiment["analysis_script"]),
            plot_script=scripts_root / "plot_results.py",
            has_plot_script=bool(experiment["has_plot_script"]),
            supports_skip_current=bool(experiment["supports_skip_current"]),
            plot_after_skip_current=bool(experiment["plot_after_skip_current"]),
            plan_title=str(experiment["plan_title"]),
            plan_description=str(experiment["plan_description"]),
        )

    return configs


def enabled_experiment_ids() -> tuple[str, ...]:
    return tuple(load_enabled_experiment_configs())


def resolve_binary(explicit_path: str | None) -> Path:
    if explicit_path:
        binary = Path(explicit_path)
        if not binary.is_absolute():
            binary = (ROOT / binary).resolve()
        if not binary.exists():
            raise FileNotFoundError(f"Benchmark binary not found: {binary}")
        return binary

    candidates = (
        ROOT / "build-tests-vs" / "Release" / "gpu_memory_layout_experiments.exe",
        ROOT / "build-tests-vs" / "Debug" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "Release" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "Debug" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "windows-x64" / "Release" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "windows-x64" / "Debug" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "gpu_memory_layout_experiments.exe",
        ROOT / "build" / "gpu_memory_layout_experiments",
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find benchmark binary. Build first, for example:\n"
        "  cmake --preset windows-tests-vs\n"
        "  cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments"
    )
