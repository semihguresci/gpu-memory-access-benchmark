# GPU Memory Layout and Cache Behavior

Repository: `gpu-memory-layout-playground`

This project is an incremental Vulkan compute benchmarking program focused on how GPUs execute code and how memory behavior affects performance.

## Purpose
The goal is to build a strong, reproducible performance portfolio that shows the ability to:
- reason about GPU execution behavior
- design architecture-aware experiments
- measure performance correctly
- connect low-level results to rendering-adjacent systems

## Project Principles
Every experiment should:
- test one primary concept
- produce quantitative output
- include short theory and clear conclusions
- be reproducible
- build on previous experiments

## Current Scope
The roadmap has been expanded to 25 experiments across six levels:
1. Foundations: execution model and Vulkan compute baselines
2. Memory layout fundamentals
3. Access patterns and cache behavior
4. On-chip memory, occupancy, and control flow
5. Atomics and parallel primitives
6. Rendering-adjacent capstone pipeline

Detailed status and per-experiment goals are tracked in:
- `docs/development_principles_and_plan.md`
- `docs/advanced_investigations_roadmap.md`
- `docs/core_experiment_plans_index.md`
- `docs/advanced_investigation_plans_index.md`

## Planned Experiment Levels
- Level 1 (01-05): dispatch basics, local size sweep, copy baseline, sequential indexing, ID mapping variants
- Level 2 (06-10): AoS vs SoA, AoSoA, std430/std140/packed, vec padding costs, scalar width sweep
- Level 3 (11-15): coalesced/strided, gather, scatter, locality reuse, bandwidth saturation
- Level 4 (16-20): shared memory tiling, tile sweep, register pressure proxy, divergence, barrier costs
- Level 5 (21-24): reduction, scan, histogram contention, stream compaction
- Level 6 (25): spatial binning or clustered culling capstone

## Advanced Extension Track
After the first 25 experiments, the project continues with 12 advanced investigations focused on:
- GPU sorting and data reordering
- BVH and traversal-oriented layout analysis
- rendering culling/list-building pipelines
- subgroup-level optimization
- async overlap and scheduling models
- occupancy-guided interpretation
- cross-GPU reproducibility

Recommended follow-up execution order:
1. Core primitives: radix sort, subgroup operations, occupancy modeling
2. Rendering data systems: tiled assignment, frustum vs clustered, GPU-driven building blocks
3. Architecture depth: BVH layouts, ray-friendly memory layouts, frame-to-frame coherence
4. Systems/platform depth: persistent work queues, async overlap, cross-GPU comparison

## Repository Layout
```text
gpu-memory-layout-playground/
|- docs/
|- experiments/
|- shaders/
|- src/
|- scripts/
|- results/
`- third_party/
```

## Benchmarking Standards
- prefer GPU timestamp queries for timing
- separate upload, dispatch, and readback timing when applicable
- include warmup passes
- report median and p95 when useful
- keep raw machine-readable outputs
- validate correctness for every experiment
- log GPU, driver, Vulkan version, OS, and build options

## Generated Artifacts Policy
Generated benchmark outputs are intentionally not committed.

Ignored artifact paths include:
- `experiments/*/results/charts/*`
- `experiments/*/results/tables/*`
- `experiments/*/runs/**/*`

This keeps the repository source-focused while still allowing full local regeneration.

## Regenerate Experiment Artifacts
`generate_experiment_artifacts.py` only builds artifacts from existing logs and does not run benchmarks.

Collect fresh benchmark data first:

```powershell
python scripts/run_experiment_data_collection.py
```

Then regenerate charts/tables:

```powershell
python scripts/generate_experiment_artifacts.py
```

Optional variants:

```powershell
python scripts/run_experiment_data_collection.py --iterations 10 --warmup 3 --size 8M --validation
python scripts/generate_experiment_artifacts.py --collect-run
```

## Per-Experiment Output Template
```text
experiments/NN_name/
|- README.md
|- shader.comp
|- host.cpp
|- config.json
|- results.csv
|- chart.png
`- notes.md
```

## Lecture-Note Planning Packs
- Core lecture-note plans (01-25): `docs/experiment_plans/`
- Advanced lecture-note plans (01-12): `docs/advanced_plans/`

## Documentation Roadmap
- `docs/README.md`: documentation landing page and reading order
- `docs/development_principles_and_plan.md`: development principles plus end-to-end program structure and milestones
- `docs/core_experiment_plans_index.md`: indexed core plans (01-25)
- `docs/experiment_plans/`: detailed lecture-note plan per core experiment
- `docs/advanced_investigations_roadmap.md`: advanced-track roadmap and phases
- `docs/advanced_investigation_plans_index.md`: indexed advanced plans (01-12)
- `docs/advanced_plans/`: detailed lecture-note plan per advanced investigation

## Portfolio Signal
By the end of the 25 experiments, this repository should show practical ability to build Vulkan compute benchmarks, analyze memory behavior, and implement performance-relevant GPU systems.
