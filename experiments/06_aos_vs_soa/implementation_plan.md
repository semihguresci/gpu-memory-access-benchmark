# Experiment 06 AoS vs SoA: Implementation Plan

Date: 2026-03-15

## Goal
Implement a correctness-first Vulkan compute experiment that compares two memory layouts on matched particle-update logic:
- `aos`: one packed struct buffer
- `soa`: eight float arrays

## Scope
- Experiment ID: `06_aos_vs_soa`
- Size sweep: preferred `1M`, `5M`, `10M` particles (fallback to smaller sizes when scratch-limited)
- Local size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms (primary), end-to-end ms (supporting)

## File Changes
New files:
- `experiments/06_aos_vs_soa/results.md`
- `experiments/06_aos_vs_soa/development_plan.md`
- `experiments/06_aos_vs_soa/implementation_plan.md`
- `experiments/06_aos_vs_soa/scripts/README.md`
- `experiments/06_aos_vs_soa/scripts/collect_run.py`
- `experiments/06_aos_vs_soa/scripts/analyze_aos_vs_soa.py`
- `experiments/06_aos_vs_soa/scripts/plot_results.py`

Touched files:
- `include/experiments/aos_soa_experiment.hpp`
- `src/experiments/aos_soa_experiment.cpp`
- `src/experiments/adapters/aos_soa_adapter.cpp`
- `scripts/run_experiment_data_collection.py`
- `scripts/generate_experiment_artifacts.py`
- `tests/unit/app_options_tests.cpp`
- `tests/unit/json_exporter_tests.cpp`

## Work Packages
### 1. Runtime Contract and Adapter
- [x] Extend experiment output to include rows and correctness aggregate.
- [x] Update adapter to pass rows and enforce correctness.

### 2. Vulkan Runtime
- [x] Keep explicit setup/teardown for AoS and SoA pipelines and buffers.
- [x] Add mapped buffer initialization and validation paths for both layouts.
- [x] Keep dispatch timing based on GPU timestamps.

### 3. Measurement and Data Export
- [x] Emit row-level metrics for each timed iteration.
- [x] Emit per-case summary statistics via `BenchmarkRunner`.
- [x] Include per-row notes for run context and failure flags.

### 4. Experiment-local Tooling
- [x] Add run collection script.
- [x] Add multi-run analysis script and output naming.
- [x] Add quick single-run plotting script.

### 5. Workflow Integration
- [x] Add experiment to `run_experiment_data_collection.py`.
- [x] Add experiment to `generate_experiment_artifacts.py`.

### 6. Verification and Quality Gates
- [ ] Build `gpu_memory_layout_experiments`.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` on modified translation units when compile database is available.
- [ ] Run unit tests.
