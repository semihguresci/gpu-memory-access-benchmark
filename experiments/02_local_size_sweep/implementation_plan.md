# Experiment 02 Local Size Sweep: Implementation Plan

Date: 2026-03-14

## Goal
Implement a correctness-first Vulkan compute local-size sweep that outputs stable per-point GPU timing metrics and a defendable local-size recommendation.

## Scope
- Experiment ID: `02_local_size_sweep`
- Local-size candidate set: `32, 64, 128, 256, 512, 1024` (device-filtered)
- Problem-size sweep: powers of two from `2^14` to `2^24` (runtime clamped)
- Variants: `contiguous_write` (primary) and `noop` (control)
- Timing outputs: dispatch GPU ms (primary), end-to-end ms (supporting)
- Validation: deterministic correctness per measured row

## Planned File Changes
New files (planned):
- `include/experiments/local_size_sweep_experiment.hpp`
- `src/experiments/local_size_sweep_experiment.cpp`
- `src/experiments/adapters/local_size_sweep_adapter.cpp`
- `shaders/02_local_size_32.comp`
- `shaders/02_local_size_64.comp`
- `shaders/02_local_size_128.comp`
- `shaders/02_local_size_256.comp`
- `shaders/02_local_size_512.comp`
- `shaders/02_local_size_1024.comp`
- `experiments/02_local_size_sweep/scripts/README.md`
- `experiments/02_local_size_sweep/scripts/analyze_local_size_sweep.py`
- `experiments/02_local_size_sweep/scripts/plot_results.py`
- `experiments/02_local_size_sweep/results.md`

Touched files (planned):
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt` (only if explicit source list needs update)
- `docs/features/features.md` (optional registration documentation update)

## Work Packages
### 1. Experiment Public Contract
- [ ] Define `LocalSizeSweepExperimentConfig` and output struct in new header.
- [ ] Keep interface consistent with existing experiment modules.
- [ ] Include only required headers in public API.

### 2. Shader Variant Assets
- [ ] Add one shader per local-size candidate with identical kernel body.
- [ ] Ensure only `layout(local_size_x = N)` differs across files.
- [ ] Validate generated `.spv` names map predictably to runtime lookup.

### 3. Vulkan Resource Lifecycle
- [ ] Implement explicit structs for pipeline/buffer resources.
- [ ] Check every Vulkan call returning `VkResult`.
- [ ] Destroy all Vulkan objects in reverse creation order.
- [ ] Reset all destroyed handles to `VK_NULL_HANDLE`.

### 4. Sweep Runtime and Correctness
- [ ] Query device limits and filter legal local-size candidates.
- [ ] Build problem-size sweep from configured scratch size and limits.
- [ ] Execute warmup and timed loops via `BenchmarkRunner`.
- [ ] Capture row-level metrics and summary samples.
- [ ] Validate readback contents for each row before marking success.

### 5. Adapter and Registry Integration
- [ ] Add adapter converting experiment output to `ExperimentRunOutput`.
- [ ] Register `02_local_size_sweep` in manifest with adapter symbol.
- [ ] Reconfigure and verify generated registry contains new descriptor.

### 6. Analysis Pipeline
- [ ] Add analysis script to produce:
- [ ] median/p95 pivot by `problem_size x local_size`
- [ ] local-size ranking table with speedup vs baseline local size 64
- [ ] Add plotting script to generate:
- [ ] median-ms sweep chart
- [ ] throughput sweep chart

### 7. Verification and Quality Gates
- [ ] Build and run with validation disabled and enabled.
- [ ] Confirm correctness failures fail the run with actionable messages.
- [ ] Run `clang-format` on touched files in `include/` and `src/`.
- [ ] Run `clang-tidy` for modified translation units when compile commands are available.
- [ ] Run `ctest` unit suite after integration.

### 8. Documentation and Reproducibility
- [ ] Update experiment README and plans if implementation decisions diverge.
- [ ] Add `results.md` with command lines, hardware metadata, and recommendation.
- [ ] Store run snapshots under `experiments/02_local_size_sweep/runs/` when collected.

## Verification Command Snapshot
Configure and build:
```powershell
cmake -S . -B build-tests-vs -G "Visual Studio 18 2026" -A x64 -DBUILD_TESTING=ON
cmake --build build-tests-vs --config Release --target gpu_memory_layout_experiments
```

Run Experiment 02:
```powershell
.\build-tests-vs\Release\gpu_memory_layout_experiments.exe --experiment 02_local_size_sweep --iterations 20 --warmup 5 --size 16M --output experiments/02_local_size_sweep/results/tables/benchmark_results
```

Run unit tests:
```powershell
ctest --test-dir build-tests-vs -C Release -L unit --output-on-failure
```

## Definition of Done
- [ ] Experiment 02 executes end-to-end through registry and CLI.
- [ ] All measured points record correctness + timing outputs.
- [ ] Illegal local sizes are skipped safely and reported explicitly.
- [ ] Exported data supports plotting and ranking local-size performance.
- [ ] Documentation includes final recommendation and reproducibility commands.
