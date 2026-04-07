# Documentation Index

This folder contains the planning, architecture, and research-positioning notes for the Vulkan compute benchmark program.

## Recommended Reading Order
1. [Research Overview](research_overview.md)
2. [Project Architecture](architecture.md)
3. [Development Principles and Plan](development_principles_and_plan.md)
4. [Implementation Plan](implementation_plan.md)
5. [Core Experiment Plans Index](core_experiment_plans_index.md)
6. [Advanced Investigations Roadmap](advanced_investigations_roadmap.md)
7. [Advanced Investigation Plans Index](advanced_investigation_plans_index.md)

## Results-First Documents
- [research_overview.md](research_overview.md): public-facing project brief, methodology, key results, and priority extensions.
- [../readme.md](../readme.md): top-level repository overview with headline findings and visuals.

## Program Plans
- [development_principles_and_plan.md](development_principles_and_plan.md): project principles, milestones, and quality gates.
- [implementation_plan.md](implementation_plan.md): concrete execution sequence for platform hardening, shared-library extraction, reporting, and validation.

## Experiment Plans
- Plan index: [core_experiment_plans_index.md](core_experiment_plans_index.md)
- Per-experiment plans: [`experiment_plans/`](experiment_plans/)
- Active implementation plan: [implementation_plan.md](implementation_plan.md)

## Priority Extensions
- [Experiment 26: Warp-Level Coalescing Alignment](experiment_plans/26_warp_level_coalescing_alignment.md)
- [Experiment 27: Cache Thrashing, Random vs Sequential](experiment_plans/27_cache_thrashing_random_vs_sequential.md)
- [Experiment 28: Device-Local vs Host-Visible Heap Placement](experiment_plans/28_device_local_vs_host_visible_heap_placement.md)
- [Experiment 29: Shared Memory Bank Conflict Study](experiment_plans/29_shared_memory_bank_conflict_study.md)
- [Experiment 30: Subgroup Reduction Variants](experiment_plans/30_subgroup_reduction_variants.md)
- [Experiment 31: Subgroup Scan Variants](experiment_plans/31_subgroup_scan_variants.md)
- [Experiment 32: Subgroup Stream Compaction Variants](experiment_plans/32_subgroup_stream_compaction_variants.md)
- [Experiment 33: 2D Locality and Transpose Study](experiment_plans/33_two_dimensional_locality_transpose_study.md)

## Advanced Plans
- Plan index: [advanced_investigation_plans_index.md](advanced_investigation_plans_index.md)
- Per-investigation plans: [`advanced_plans/`](advanced_plans/)

## Tooling Notes
- [clang_tooling.md](clang_tooling.md)
- [layout_rules.md](layout_rules.md)

## Testing Notes
- [tests/README.md](tests/README.md)
- [tests/test_plan.md](tests/test_plan.md)
- [tests/test_coverage.md](tests/test_coverage.md)

## Feature Notes
- [features/features.md](features/features.md)
- [features/experiment_registry_generation.md](features/experiment_registry_generation.md)
