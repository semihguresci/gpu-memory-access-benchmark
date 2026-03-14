# Documentation Index

This folder contains the planning and reference notes for the Vulkan compute benchmark roadmap.

## Recommended Reading Order
1. [Project Architecture](architecture.md)
2. [Development Principles and Plan](development_principles_and_plan.md)
3. [Core Experiment Plans Index (01-25)](core_experiment_plans_index.md)
4. [Advanced Investigations Roadmap](advanced_investigations_roadmap.md)
5. [Advanced Investigation Plans Index (01-12)](advanced_investigation_plans_index.md)

## Core Plans (01-25)
- Plan index: [core_experiment_plans_index.md](core_experiment_plans_index.md)
- Per-experiment plans: [`experiment_plans/`](experiment_plans/)
- Active implementation plan: [../experiments/01_dispatch_basics/development_plan.md](../experiments/01_dispatch_basics/development_plan.md)

## Advanced Plans (01-12)
- Plan index: [advanced_investigation_plans_index.md](advanced_investigation_plans_index.md)
- Per-investigation plans: [`advanced_plans/`](advanced_plans/)

## Core Topic Coverage
- Execution and dispatch fundamentals: Experiments 01-05 in [`experiment_plans/`](experiment_plans/)
- Memory layout and alignment: Experiments 06-10 in [`experiment_plans/`](experiment_plans/)
- Access pattern and cache behavior: Experiments 11-15 in [`experiment_plans/`](experiment_plans/)
- On-chip memory and control flow: Experiments 16-20 in [`experiment_plans/`](experiment_plans/)
- Parallel primitives and capstone integration: Experiments 21-25 in [`experiment_plans/`](experiment_plans/)

## Tooling Notes
- [clang_tooling.md](clang_tooling.md)
- [layout_rules.md](layout_rules.md)

## Feature Notes
- [features/features.md](features/features.md)
- [features/experiment_registry_generation.md](features/experiment_registry_generation.md)
