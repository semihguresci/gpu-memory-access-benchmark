#!/usr/bin/env python3

from __future__ import annotations

from experiment_manifest import ROOT, load_experiment_manifest


OUTPUT_PATH = ROOT / "docs" / "core_experiment_plans_index.md"


def _memory_sequence(experiments: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        [experiment for experiment in experiments if experiment.get("memory_sequence_order") is not None],
        key=lambda experiment: int(experiment["memory_sequence_order"]),
    )


def _core_experiments(experiments: list[dict[str, object]]) -> list[dict[str, object]]:
    return [experiment for experiment in experiments if experiment.get("docs_group") == "core"]


def _extension_experiments(experiments: list[dict[str, object]]) -> list[dict[str, object]]:
    return [experiment for experiment in experiments if experiment.get("docs_group") == "extension"]


def _format_line(experiment: dict[str, object]) -> str:
    experiment_number = str(experiment["id"])[:2]
    return (
        f"- [Experiment {experiment_number}: {experiment['plan_title']}]"
        f"({experiment['plan_path']}): {experiment['plan_description']}"
    )


def main() -> None:
    experiments = load_experiment_manifest()

    lines = [
        "<!-- Generated from config/experiment_manifest.json via scripts/generate_experiment_plan_index.py -->",
        "# Core Experiment Plans Index (Lecture Notes Track)",
        "",
        "This index points to the detailed lecture-note style plan for each core experiment.",
        "",
        "## Recommended Reading Order",
        "- Start with Experiments 01-05 to stabilize benchmark methodology and execution-model intuition.",
        "- Continue with Experiments 06-10 for layout and alignment design rules.",
        "- Use Experiments 11-15 to map access pattern, cache behavior, and saturation.",
        "- Use Experiments 16-20 to build architecture-aware optimization intuition.",
        "- Finish with Experiments 21-25 to assemble practical parallel primitives and capstone systems.",
        "",
        "## Memory Optimization Sequence",
    ]

    for experiment in _memory_sequence(experiments):
        lines.append(
            f"- [Experiment {str(experiment['id'])[:2]}: {experiment['plan_title']}]({experiment['plan_path']})"
        )

    lines.extend(["", "## Core Experiment Plans"])
    for experiment in _core_experiments(experiments):
        lines.append(_format_line(experiment))

    lines.extend(["", "## Priority Extensions Beyond Core 25"])
    for experiment in _extension_experiments(experiments):
        lines.append(_format_line(experiment))

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
