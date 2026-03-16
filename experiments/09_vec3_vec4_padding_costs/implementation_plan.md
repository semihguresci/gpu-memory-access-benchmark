# Experiment 09 vec3 vs vec4 Padding Costs: Implementation Plan

Date: 2026-03-16

## Goal
Implement a correctness-first Vulkan compute experiment that compares three representations for the same logical particle payload:
- `vec3_padded` (alignment-heavy AoS)
- `vec4` (packed AoS using `w` lanes)
- `split_scalars` (SoA scalar arrays)

Logical payload per particle:
- `11` floats (`coeffs[3]`, `position[3]`, `velocity[3]`, `mass`, `dt`) => `44` logical bytes

## Scope
- Experiment ID: `09_vec3_vec4_padding_costs`
- Size sweep: preferred `131072`, `262144`, `524288`, `1048576` (or smaller if scratch-limited)
- Local size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms (primary), end-to-end ms (supporting)

## File Changes
New files:
- `include/experiments/vec3_vec4_padding_costs_experiment.hpp`
- `src/experiments/vec3_vec4_padding_costs_experiment.cpp`
- `src/experiments/adapters/vec3_vec4_padding_costs_adapter.cpp`
- `shaders/09_vec3_vec4_padding_costs/09_vec3_padded.comp`
- `shaders/09_vec3_vec4_padding_costs/09_vec4.comp`
- `shaders/09_vec3_vec4_padding_costs/09_split_scalars.comp`
- `experiments/09_vec3_vec4_padding_costs/architecture.md`
- `experiments/09_vec3_vec4_padding_costs/development_plan.md`
- `experiments/09_vec3_vec4_padding_costs/implementation_plan.md`
- `experiments/09_vec3_vec4_padding_costs/results.md`
- `experiments/09_vec3_vec4_padding_costs/scripts/collect_run.py`
- `experiments/09_vec3_vec4_padding_costs/scripts/README.md`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`

## Work Packages
### 1. Runtime Contract and Adapter
- [x] Add Experiment 09 config/output contract.
- [x] Add adapter wiring and correctness enforcement.

### 2. Vulkan Runtime
- [x] Add per-variant buffer/pipeline setup and teardown.
- [x] Add deterministic host seed/validation paths for all variants.
- [x] Keep dispatch timing based on GPU timestamps.

### 3. Measurement and Export
- [x] Emit row-level metrics for each timed iteration.
- [x] Emit per-case summary metrics via `BenchmarkRunner`.
- [x] Include layout/waste context in row notes.

### 4. Workflow Integration
- [x] Add experiment to registry manifest and build target list.
- [x] Add experiment to root data collection helper.
- [x] Add experiment-local run collection script.

### 5. Verification Gates
- [x] Build Debug profile (`windows-tests-vs`).
- [x] Run `clang-format` on touched C++ files.
- [x] Run `clang-tidy` for touched translation units.
- [x] Run `ctest` suite.
- [x] Run Experiment 09 benchmark and collect run artifacts.
