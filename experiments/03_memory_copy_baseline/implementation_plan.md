# Experiment 03 Memory Copy Baseline: Implementation Plan

Date: 2026-03-15

## Goal
Implement a correctness-first Vulkan compute memory throughput baseline with three isolated memory modes:
- read-only
- write-only
- read+write copy

## Scope
- Experiment ID: `03_memory_copy_baseline`
- Size sweep: powers of two from `1 MiB` to `1 GiB` (runtime-clamped)
- Fixed local size: `256`
- Dispatch count: `1`
- Timing outputs: dispatch GPU ms (primary), end-to-end ms (supporting)

## File Changes
New files:
- `include/experiments/memory_copy_baseline_experiment.hpp`
- `src/experiments/memory_copy_baseline_experiment.cpp`
- `src/experiments/adapters/memory_copy_baseline_adapter.cpp`
- `shaders/03_memory_read_only.comp`
- `shaders/03_memory_write_only.comp`
- `shaders/03_memory_read_write_copy.comp`
- `experiments/03_memory_copy_baseline/architecture.md`
- `experiments/03_memory_copy_baseline/development_plan.md`
- `experiments/03_memory_copy_baseline/implementation_plan.md`
- `experiments/03_memory_copy_baseline/results.md`
- `experiments/03_memory_copy_baseline/scripts/README.md`
- `experiments/03_memory_copy_baseline/scripts/collect_run.py`
- `experiments/03_memory_copy_baseline/scripts/analyze_memory_copy_baseline.py`
- `experiments/03_memory_copy_baseline/scripts/plot_results.py`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`
- `scripts/generate_experiment_artifacts.py`
- `experiments/03_memory_copy_baseline/README.md`

## Work Packages
### 1. Runtime Contract and Entry Points
- [x] Add public experiment contract header.
- [x] Add adapter integration for generated registry.
- [x] Register experiment manifest entry.

### 2. Vulkan Runtime
- [x] Implement explicit resource structs for buffers and pipelines.
- [x] Check all Vulkan calls returning `VkResult`.
- [x] Destroy Vulkan resources in reverse creation order.
- [x] Reset destroyed handles to `VK_NULL_HANDLE`.

### 3. Mode Execution and Validation
- [x] Implement read-only path.
- [x] Implement write-only path.
- [x] Implement read+write copy path.
- [x] Add deterministic correctness checks per mode.
- [x] Add per-row notes and correctness flags.

### 4. Measurement and Data Export
- [x] Collect dispatch GPU timing via timestamp queries.
- [x] Collect end-to-end timing for supporting context.
- [x] Emit row-level throughput and mode-specific GB/s.
- [x] Summarize each case with median/p95 support through `BenchmarkRunner`.

### 5. Experiment-local Tooling
- [x] Add run collection script.
- [x] Add analysis script for CSV/chart generation.
- [x] Add quick-plot script from current benchmark JSON.

### 6. Verification and Quality Gates
- [ ] Build `gpu_memory_layout_experiments`.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` on modified translation units when compile database is available.
- [ ] Run tests (`ctest`) for regressions.

## Run Commands
Build:
```powershell
cmake --build build-tests-vs --config Release --target gpu_memory_layout_experiments
```

Collect experiment data:
```powershell
python scripts/run_experiment_data_collection.py --experiment 03_memory_copy_baseline --iterations 10 --warmup 3 --size 64M
```

Generate experiment-local artifacts:
```powershell
python scripts/generate_experiment_artifacts.py --experiment 03_memory_copy_baseline --collect-run
```
