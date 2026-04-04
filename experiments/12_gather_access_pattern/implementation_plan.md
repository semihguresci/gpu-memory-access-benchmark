# Experiment 12 Gather Access Pattern: Implementation Plan

Date: 2026-03-25

## Goal
Implement a correctness-first Vulkan compute experiment that sweeps indirect gather distributions while keeping arithmetic, element type, and logical element count constant.

Primary configuration:
- Experiment ID: `12_gather_access_pattern`
- Distribution sweep: `identity`, `block_coherent_32`, `clustered_random_256`, `random_permutation`
- Workgroup size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms primary, end-to-end ms supporting
- Buffer budget: derive physical allocation from the existing scratch-size option so the largest logical count fits without wraparound
- Accuracy outputs: deterministic CPU reference for gathered destination values; padding must not affect correctness checks

## Scope
- Run target: `identity` versus `block_coherent_32`, `clustered_random_256`, and `random_permutation`
- Logical element count: fixed per sweep, with physical buffer span equal across variants
- Reporting: per-distribution median and p95 timing, logical throughput in elements/s, and GB/s based on logical bytes moved
- Validation policy: fail the run on any mismatch before emitting benchmark success

## File Changes
New files:
- `include/experiments/gather_access_pattern_experiment.hpp`
- `src/experiments/gather_access_pattern_experiment.cpp`
- `src/experiments/adapters/gather_access_pattern_adapter.cpp`
- `shaders/12_gather_access_pattern/12_gather_access_pattern.comp`
- `experiments/12_gather_access_pattern/scripts/collect_run.py`
- `experiments/12_gather_access_pattern/scripts/analyze_gather_access_pattern.py`
- `experiments/12_gather_access_pattern/scripts/plot_results.py`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`
- `scripts/generate_experiment_artifacts.py`
- `experiments/12_gather_access_pattern/README.md`

## Work Packages
### 1. Runtime Contract and Kernel
- [x] Add Experiment 12 config/output contract.
- [x] Add deterministic index-pattern generation for the chosen distributions.
- [x] Add CPU reference generation for the logical destination order.
- [x] Keep dispatch timing based on GPU timestamps.

### 2. Vulkan Runtime
- [x] Add buffer, descriptor, and pipeline setup/teardown for the gather kernel.
- [x] Keep address math explicit and bounds-safe for every distribution.
- [x] Preserve exact comparison semantics for gathered destination values.

### 3. Correctness and Metrics
- [x] Validate the gather path against CPU reference outputs.
- [x] Emit row-level metrics for each timed iteration and distribution.
- [x] Capture logical bytes processed so GB/s stays comparable across distributions.

### 4. Workflow Integration
- [x] Add experiment to the manifest and build target list.
- [x] Add experiment to the root data-collection helper.
- [x] Add experiment to the artifact-generation helper.
- [x] Add experiment-local run collection script.
- [x] Add experiment-local analysis script.

### 5. Verification Gates
- [x] Build release target (`tests-vs-release` preset).
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units.
- [x] Run tests and a benchmark smoke pass for Experiment 12.
