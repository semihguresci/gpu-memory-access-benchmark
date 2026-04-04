# Experiment 16 Shared Memory Tiling: Implementation Plan

Date: 2026-03-27

## Goal
Implement a correctness-first Vulkan compute experiment that compares a direct global-memory stencil against an explicitly tiled shared-memory stencil while keeping arithmetic, workgroup size, and logical output count constant.

Primary configuration:
- Experiment ID: `16_shared_memory_tiling`
- Variant set: `direct_global`, `shared_tiled`
- Reuse-radius sweep: `1`, `4`, `8`, `16`
- Workgroup size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms primary, end-to-end ms supporting
- Buffer layout: one padded source buffer plus one output buffer; size the padded source for the max radius so every radius shares the same logical output count
- Accuracy outputs: deterministic CPU stencil reference with exact `uint32_t` comparison for every output element
- Measurement note: keep tile size fixed in this experiment; Experiment 17 will vary tile size using the same tiled kernel contract

## Variant Contract
- `direct_global`: each invocation reads its full `2 * radius + 1` neighborhood from the source buffer in device memory and writes one output sum
- `shared_tiled`: each workgroup stages `local_size_x + 2 * radius` source elements into workgroup memory, executes one barrier, and computes the same output sum from staged data

All primary variants must produce byte-identical outputs for the same radius and logical size.

## Scope
- Run target: direct-versus-tiled comparison at fixed `local_size_x = 256`
- Logical output count: one steady-state size derived from `--size`, with buffer budgeting based on the max-radius padded source plus output span
- Reporting: per-variant and per-radius median and p95 timing, stencil outputs per second in `throughput`, estimated global GB/s in `gbps`, and tiling metadata in row notes
- Validation policy: fail the run on any output mismatch or non-finite dispatch timing before emitting benchmark success

## File Changes
New files:
- `include/experiments/shared_memory_tiling_experiment.hpp`
- `src/experiments/shared_memory_tiling_experiment.cpp`
- `src/experiments/adapters/shared_memory_tiling_adapter.cpp`
- `shaders/16_shared_memory_tiling/16_shared_memory_tiling_direct.comp`
- `shaders/16_shared_memory_tiling/16_shared_memory_tiling_tiled.comp`
- `experiments/16_shared_memory_tiling/architecture.md`
- `experiments/16_shared_memory_tiling/results.md`
- `experiments/16_shared_memory_tiling/scripts/collect_run.py`
- `experiments/16_shared_memory_tiling/scripts/analyze_shared_memory_tiling.py`
- `experiments/16_shared_memory_tiling/scripts/plot_results.py`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`
- `scripts/generate_experiment_artifacts.py`
- `experiments/16_shared_memory_tiling/README.md`

## Work Packages
### 1. Runtime Contract and Data Layout
- [ ] Add Experiment 16 config/output contract.
- [ ] Define the padded source layout using `max_radius = 16` so all variants share one logical output span.
- [ ] Generate deterministic bounded `uint32_t` source values and known-pattern output initialization.
- [ ] Record row-note metadata such as `reuse_radius`, `tile_span_elements`, `shared_bytes_per_workgroup`, `barriers_per_workgroup`, `estimated_global_read_bytes`, `local_size_x`, and `group_count_x`.

### 2. Vulkan Runtime and Shader Paths
- [ ] Add buffer, descriptor, and pipeline setup/teardown for both shader variants.
- [ ] Resolve and load `16_shared_memory_tiling_direct.comp.spv` and `16_shared_memory_tiling_tiled.comp.spv` through the existing shader-path helper.
- [ ] Keep the direct kernel explicit and bounds-safe over the padded source span.
- [ ] Keep the tiled kernel explicit: cooperative halo loading, exactly one workgroup barrier, then neighborhood accumulation from workgroup memory.
- [ ] Preserve reverse-order teardown and handle reset rules for every Vulkan object.

### 3. Correctness and Metrics
- [ ] Build a CPU reference implementation for every radius over the padded source data.
- [ ] Validate every output element after each timed dispatch.
- [ ] Emit row-level metrics for each `variant x radius` timed sample.
- [ ] Use `throughput` for stencil outputs per second and `gbps` for estimated global bytes per second, with the calculation mode documented in row notes.
- [ ] Summarize timing samples with the existing benchmark runner utilities.

### 4. Workflow Integration
- [ ] Add Experiment 16 to the manifest and build target list.
- [ ] Add Experiment 16 to the root data-collection helper.
- [ ] Add Experiment 16 to the root artifact-generation helper.
- [ ] Add experiment-local run collection, analysis, and plotting scripts.
- [ ] Update experiment-local reporting scaffolding so charts and tables have deterministic names.

### 5. Forward Compatibility with Experiment 17
- [ ] Keep `local_size_x` and tile-span calculations factored cleanly so Experiment 17 can sweep tile size without rewriting validation or analysis plumbing.
- [ ] Avoid baking radius-specific constants into host-side output parsing; only the shader interface should vary by radius.
- [ ] Keep variant naming and note fields stable so Experiment 17 can compare against the Experiment 16 direct baseline.

### 6. Verification Gates
- [ ] Build release target (`tests-vs-release` preset).
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units.
- [ ] Run tests and a benchmark smoke pass for Experiment 16.
