# Experiment 13 Scatter Access Pattern: Implementation Plan

Date: 2026-03-26

## Goal
Implement a correctness-first Vulkan compute experiment that sweeps indirect scatter target distributions while keeping one atomic increment per logical input and a fixed destination capacity, so contention remains the primary changing variable.

Primary configuration:
- Experiment ID: `13_scatter_access_pattern`
- Distribution sweep: `unique_permutation`, `random_collision_x4`, `clustered_hotset_32`
- Workgroup size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms primary, end-to-end ms supporting
- Buffer budget: destination counter span stays fixed to the logical element count; contention comes from target reuse rather than resizing the destination buffer
- Accuracy outputs: deterministic CPU reference counter histogram with exact-match validation over the full destination span
- Measurement note: `throughput` is the primary rate metric; `gbps` is retained only as a logical traffic proxy and should not be treated as physical atomic bandwidth

## Variant Contract
- `unique_permutation`: every logical input targets a unique shuffled destination, preserving scatter address disorder without collisions
- `random_collision_x4`: target assignments are shuffled over a reduced active target set so each active destination receives four writers on average
- `clustered_hotset_32`: target assignments are concentrated into small hot windows to force strong localized contention

All primary variants use the same atomic scatter kernel so correctness remains defined and the comparison stays focused on target distribution.

## Scope
- Run target: unique-target scatter baseline versus low-collision and high-collision contention variants
- Logical element count: fixed per sweep, with destination capacity and index-buffer length fixed across variants
- Reporting: per-distribution median and p95 timing, update throughput, logical GB/s proxy, and contention metadata in row notes
- Validation policy: fail the run on any counter mismatch before emitting benchmark success

## File Changes
New files:
- `include/experiments/scatter_access_pattern_experiment.hpp`
- `src/experiments/scatter_access_pattern_experiment.cpp`
- `src/experiments/adapters/scatter_access_pattern_adapter.cpp`
- `shaders/13_scatter_access_pattern/13_scatter_access_pattern.comp`
- `experiments/13_scatter_access_pattern/architecture.md`
- `experiments/13_scatter_access_pattern/results.md`
- `experiments/13_scatter_access_pattern/scripts/collect_run.py`
- `experiments/13_scatter_access_pattern/scripts/analyze_scatter_access_pattern.py`
- `experiments/13_scatter_access_pattern/scripts/plot_results.py`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`
- `scripts/generate_experiment_artifacts.py`
- `experiments/13_scatter_access_pattern/README.md`

## Work Packages
### 1. Runtime Contract and Target Generation
- [ ] Add Experiment 13 config/output contract.
- [ ] Add deterministic target generation for the planned scatter distributions with a fixed seed.
- [ ] Keep destination capacity equal to the logical element count for every variant.
- [ ] Record row-note metadata such as `distribution`, `pattern_seed`, `target_capacity`, `hot_target_count`, `collision_factor`, `local_size_x`, and `group_count_x`.

### 2. Vulkan Runtime and Shader
- [ ] Add buffer, descriptor, and pipeline setup/teardown for the scatter kernel.
- [ ] Add a destination counter buffer that is reset before every warmup and timed iteration.
- [ ] Resolve and load `13_scatter_access_pattern.comp.spv` through the existing shader-path helper.
- [ ] Keep the kernel path explicit and bounds-safe: load target index, check bounds, then issue `atomicAdd`.
- [ ] Preserve reverse-order teardown and handle reset rules for all Vulkan objects.

### 3. Correctness and Metrics
- [ ] Build a CPU reference histogram for the full destination span.
- [ ] Validate every destination counter after each timed dispatch.
- [ ] Emit row-level metrics for each timed iteration and distribution.
- [ ] Capture logical bytes processed so the derived `gbps` column stays comparable as a proxy across variants.
- [ ] Summarize timing samples with the existing benchmark runner utilities.

### 4. Workflow Integration
- [ ] Add Experiment 13 to the manifest and build target list.
- [ ] Add Experiment 13 to the root data-collection helper.
- [ ] Add Experiment 13 to the root artifact-generation helper.
- [ ] Add experiment-local run collection, analysis, and plotting scripts.
- [ ] Update experiment-local reporting scaffolding so charts and tables have deterministic names.

### 5. Verification Gates
- [ ] Build release target (`tests-vs-release` preset).
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units.
- [ ] Run tests and a benchmark smoke pass for Experiment 13.
