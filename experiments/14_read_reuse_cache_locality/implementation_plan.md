# Experiment 14 Read Reuse and Cache Locality: Implementation Plan

Date: 2026-03-27

## Goal
Implement a correctness-first Vulkan compute experiment that sweeps deterministic pair-reuse schedules while keeping logical read count, reuse count, arithmetic, and destination write pattern constant.

Primary configuration:
- Experiment ID: `14_read_reuse_cache_locality`
- Reuse schedule sweep: `reuse_distance_1`, `reuse_distance_32`, `reuse_distance_256`, `reuse_distance_4096`, `reuse_distance_full_span`
- Workgroup size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms primary, end-to-end ms supporting
- Buffer budget: derive `source_count = logical_count / 2` and keep `index_count = destination_count = logical_count`, with each logical buffer checked against the existing `--size` cap
- Accuracy outputs: deterministic CPU reference for destination values with exact validation of both destination contents and the unchanged index schedule
- Measurement note: `throughput` is the primary rate metric in logical reads per second; `gbps` is retained as a logical traffic proxy based on index read + source read + destination write

## Variant Contract
- `reuse_distance_1`: schedule `[0, 0, 1, 1, 2, 2, ...]`
- `reuse_distance_32`: schedule 32 unique indices, then immediately replay the same 32 indices
- `reuse_distance_256`: schedule 256 unique indices, then immediately replay the same 256 indices
- `reuse_distance_4096`: schedule 4096 unique indices, then immediately replay the same 4096 indices
- `reuse_distance_full_span`: one full sequential pass over the unique source span followed by a second full sequential pass

All primary variants use the same gather-style kernel, keep `pair_reuse_count = 2`, and preserve sequential destination writes.

## Scope
- Run target: adjacent reuse versus medium-, far-, and full-span reuse-distance variants
- Logical invocation count: fixed per sweep, with unique source count fixed to half the logical count across all primary variants
- Reporting: per-variant median and p95 timing, read throughput, logical GB/s proxy, and reuse metadata in row notes
- Validation policy: fail the run on any mismatch before emitting benchmark success

## File Changes
New files:
- `include/experiments/read_reuse_cache_locality_experiment.hpp`
- `src/experiments/read_reuse_cache_locality_experiment.cpp`
- `src/experiments/adapters/read_reuse_cache_locality_adapter.cpp`
- `shaders/14_read_reuse_cache_locality/14_read_reuse_cache_locality.comp`
- `experiments/14_read_reuse_cache_locality/architecture.md`
- `experiments/14_read_reuse_cache_locality/development_plan.md`
- `experiments/14_read_reuse_cache_locality/implementation_plan.md`
- `experiments/14_read_reuse_cache_locality/results.md`
- `experiments/14_read_reuse_cache_locality/scripts/collect_run.py`
- `experiments/14_read_reuse_cache_locality/scripts/analyze_read_reuse_cache_locality.py`
- `experiments/14_read_reuse_cache_locality/scripts/plot_results.py`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`
- `scripts/generate_experiment_artifacts.py`
- `docs/experiment_plans/14_read_reuse_cache_locality.md`
- `experiments/14_read_reuse_cache_locality/experiment_plan.md`
- `experiments/14_read_reuse_cache_locality/README.md`

## Work Packages
### 1. Runtime Contract and Schedule Generation
- [ ] Add Experiment 14 config/output contract.
- [ ] Add deterministic pair-schedule generation for the planned reuse distances with exact tail handling for partial blocks.
- [ ] Keep source capacity fixed to `logical_count / 2` for every primary variant.
- [ ] Record row-note metadata such as `reuse_distance_reads`, `pair_block_size`, `pair_reuse_count`, `source_unique_elements`, `source_span_bytes`, `local_size_x`, and `group_count_x`.

### 2. Vulkan Runtime and Shader
- [ ] Add buffer, descriptor, and pipeline setup/teardown for the reuse-locality kernel.
- [ ] Add a source buffer, reuse-schedule index buffer, and destination buffer that are seeded before warmup and timed iterations.
- [ ] Resolve and load `14_read_reuse_cache_locality.comp.spv` through the existing shader-path helper.
- [ ] Keep the kernel path explicit and bounds-safe: load scheduled source index, check bounds, read source value, then write deterministic output.
- [ ] Preserve reverse-order teardown and handle reset rules for all Vulkan objects.

### 3. Correctness and Metrics
- [ ] Build a CPU reference destination array from the exact generated schedule.
- [ ] Validate the destination values and verify the source/index buffers remain unchanged after dispatch.
- [ ] Emit row-level metrics for each timed iteration and reuse-distance variant.
- [ ] Capture logical bytes processed so the derived `gbps` column stays comparable across variants.
- [ ] Summarize timing samples with the existing benchmark runner utilities.

### 4. Workflow Integration
- [ ] Add Experiment 14 to the manifest and build target list.
- [ ] Add Experiment 14 to the root data-collection helper.
- [ ] Add Experiment 14 to the root artifact-generation helper.
- [ ] Add experiment-local run collection, analysis, and plotting scripts.
- [ ] Update experiment-local reporting scaffolding so charts and tables have deterministic names.

### 5. Verification Gates
- [ ] Build release target (`tests-vs-release` preset).
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units.
- [ ] Run tests and a benchmark smoke pass for Experiment 14.
