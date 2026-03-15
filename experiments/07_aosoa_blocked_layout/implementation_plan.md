# Experiment 07 AoSoA Blocked Layout: Implementation Plan

Date: 2026-03-15

## Goal
Implement a correctness-first Vulkan compute experiment that compares memory layouts on matched particle-update logic:
- `aos`: one packed struct buffer
- `soa`: sixteen float arrays
- `aosoa_b4`, `aosoa_b8`, `aosoa_b16`, `aosoa_b32`: blocked hybrid layout in one storage buffer
- payload: 16 floats per particle (12 hot fields used in kernel, 4 cold fields for layout pressure)

## Scope
- Experiment ID: `07_aosoa_blocked_layout`
- Size sweep: preferred `1M`, `2M`, `4M`, `8M`, `16M`, `32M` particles (fallback to smaller sizes when scratch-limited)
- Local size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms (primary), end-to-end ms (supporting)

## File Changes
New files:
- `include/experiments/aosoa_blocked_layout_experiment.hpp`
- `src/experiments/aosoa_blocked_layout_experiment.cpp`
- `src/experiments/adapters/aosoa_blocked_layout_adapter.cpp`
- `shaders/07_aos.comp`
- `shaders/07_soa.comp`
- `shaders/07_aosoa_blocked.comp`
- `experiments/07_aosoa_blocked_layout/architecture.md`
- `experiments/07_aosoa_blocked_layout/development_plan.md`
- `experiments/07_aosoa_blocked_layout/implementation_plan.md`
- `experiments/07_aosoa_blocked_layout/results.md`
- `experiments/07_aosoa_blocked_layout/scripts/README.md`
- `experiments/07_aosoa_blocked_layout/scripts/collect_run.py`
- `experiments/07_aosoa_blocked_layout/scripts/analyze_aosoa_blocked_layout.py`
- `experiments/07_aosoa_blocked_layout/scripts/plot_results.py`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `experiments/07_aosoa_blocked_layout/README.md`
- `scripts/run_experiment_data_collection.py`
- `scripts/generate_experiment_artifacts.py`

## Work Packages
### 1. Runtime Contract and Adapter
- [x] Add experiment config/output contract for 07.
- [x] Add adapter wiring and correctness enforcement.

### 2. Vulkan Runtime
- [x] Add explicit setup/teardown for AoS, SoA, and AoSoA pipelines and buffers.
- [x] Add mapped buffer initialization and validation paths for all variants.
- [x] Keep dispatch timing based on GPU timestamps.

### 3. Measurement and Data Export
- [x] Emit row-level metrics for each timed iteration and variant.
- [x] Emit per-case summary statistics via `BenchmarkRunner`.
- [x] Include per-row notes for run context and variant configuration.

### 4. Experiment-local Tooling
- [x] Add run collection script.
- [x] Add multi-run analysis script and output naming.
- [x] Add quick single-run plotting script.

### 5. Workflow Integration
- [x] Add experiment to `run_experiment_data_collection.py`.
- [x] Add experiment to `generate_experiment_artifacts.py`.

### 6. Verification and Quality Gates
- [x] Build `gpu_memory_layout_experiments`.
- [x] Run `clang-format` on touched C++ files.
- [x] Run `clang-tidy` on modified translation units when compile database is available.
- [x] Run unit tests.
