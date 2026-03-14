# Generated Experiment Registry Architecture

## Purpose
This document defines the build-time generated experiment registry architecture (option 2) to replace hardcoded per-experiment `if` blocks in `main.cpp`.

The goal is to support many experiments without continuously editing control-flow code.

## Problem Summary
Current experiment dispatch is hardcoded:
- experiment IDs are listed in CLI validation
- experiment execution lives in manual `if` blocks
- output merge and failure handling are duplicated per experiment

As the number of experiments grows, this creates:
- frequent merge conflicts
- high risk of forgotten registration
- growing maintenance overhead in entrypoint code

## Target Architecture
Use a generated registry as the source of truth:
1. A manifest file declares experiment metadata and adapter symbol names.
2. CMake generates `experiment_registry.hpp/.cpp` from the manifest.
3. `main.cpp` iterates over the registry to execute selected experiments.
4. CLI validation resolves experiment IDs against the generated registry.

This keeps runtime code simple and deterministic while avoiding plugin complexity.

## Design Principles
- Deterministic and explicit (no runtime reflection, no hidden globals).
- One adapter per experiment to normalize output shape.
- Generated code must be stable and reviewable.
- Duplicate IDs fail fast during generation.
- Vulkan ownership remains local to each experiment implementation.

## Unified Runtime Contract
All experiments execute through one contract:

```cpp
struct ExperimentRunOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool success = true;
    std::string error_message;
};

using ExperimentRunFn = bool (*)(
    VulkanContext& context,
    const BenchmarkRunner& runner,
    const AppOptions& options,
    ExperimentRunOutput& output);

struct ExperimentDescriptor {
    std::string_view id;
    std::string_view display_name;
    std::string_view category;
    bool enabled;
    ExperimentRunFn run;
};
```

## Build-Time Generation Model
Recommended manifest format: simple CMake list file or JSON read by CMake script.

Minimum fields per experiment:
- `id` (example: `01_dispatch_basics`)
- `display_name`
- `category` (example: `foundations`, `memory_layout`)
- `adapter_symbol` (function symbol used in generated registry)
- `enabled` (boolean)

Generated outputs:
- `build/generated/experiment_registry.hpp`
- `build/generated/experiment_registry.cpp`

`experiment_registry.cpp` contains:
- static descriptor array
- helper lookup functions:
  - `find_experiment_by_id(...)`
  - `enabled_experiments()`
  - `all_experiment_ids()`

## Project File Plan
New files:
- `include/experiments/experiment_contract.hpp`
- `include/experiments/experiment_registry.hpp` (public declaration)
- `src/experiments/experiment_registry.cpp.in` (template)
- `cmake/experiments_manifest.cmake` (manifest data)
- `cmake/generate_experiment_registry.cmake` (validation and rendering)
- `src/experiments/adapters/dispatch_basics_adapter.cpp`
- `src/experiments/adapters/aos_soa_adapter.cpp`

Touched files:
- `src/main.cpp` (replace manual `if` dispatch with generic loop)
- `src/utils/app_options.cpp` (remove hardcoded `IsMember` list)
- `CMakeLists.txt` (add generation step and include generated directory)

## Implementation Tasks

### Task 1: Introduce Contract Types
Steps:
1. Add `ExperimentRunOutput`, `ExperimentRunFn`, and `ExperimentDescriptor`.
2. Keep contract header lightweight and include-minimal.
3. Ensure warning-clean build.

Deliverable:
- Contract compiles without changing runtime behavior.

### Task 2: Add Adapter Layer
Steps:
1. Create one adapter function for each existing experiment.
2. Map experiment-specific output to `ExperimentRunOutput`.
3. Keep experiment-specific correctness checks in adapter.

Deliverable:
- Existing experiments can run through unified API.

### Task 3: Create Manifest
Steps:
1. Add manifest file with one entry per experiment.
2. Include `id`, `display_name`, `category`, `adapter_symbol`, `enabled`.
3. Add unique-ID validation logic.

Deliverable:
- Build fails with clear error for duplicate or malformed entries.

### Task 4: Generate Registry Code in CMake
Steps:
1. Add CMake script that reads manifest and assembles descriptor entries.
2. Generate `.hpp/.cpp` from `.in` templates using `configure_file`.
3. Add generated sources to benchmark target.
4. Add generated include directory to target include paths.

Deliverable:
- Registry compiles from generated files without manual edits.

### Task 5: Refactor `main.cpp` to Generic Dispatch
Steps:
1. Parse selected experiment IDs (`all` or comma-separated IDs).
2. Resolve IDs against generated registry.
3. Execute each descriptor via `run(...)`.
4. Merge `summary_results` and `rows` uniformly.
5. Keep existing shutdown and error behavior explicit.

Deliverable:
- No hardcoded per-experiment `if` chain in main.

### Task 6: Refactor CLI Validation
Steps:
1. Remove static `IsMember({ ... })`.
2. Accept string input and validate IDs against registry after parse.
3. Print available IDs on invalid input.

Deliverable:
- CLI auto-supports new manifest entries.

### Task 7: Verification and Tooling
Steps:
1. Build clean on MSVC and Clang/GCC warning flags.
2. Run at least:
   - `--experiment 01_dispatch_basics`
   - `--experiment 06_aos_vs_soa`
   - `--experiment all`
3. Confirm JSON output still contains expected summary and row records.
4. Run formatting and `clang-tidy` on touched translation units.

Deliverable:
- Functional parity with current behavior and cleaner extensibility.

## Rollout Plan
1. Land infrastructure with only current experiments registered.
2. Migrate new experiments only through manifest + adapter path.
3. Enforce policy in review: no direct experiment branching in `main.cpp`.

## Definition of Done
- Adding a new experiment requires:
  1. new experiment implementation
  2. new adapter function
  3. one manifest entry
- No changes needed in main dispatch logic.
- CLI accepts new experiment ID without hardcoded list edits.
- Build fails clearly on duplicate experiment IDs.

## Risks and Mitigations
- Risk: generated code becomes opaque.
  - Mitigation: keep templates small and checked into source.
- Risk: adapter drift from experiment output.
  - Mitigation: adapters remain thin, with focused tests where practical.
- Risk: accidental disabling in manifest.
  - Mitigation: print enabled/disabled state at startup when verbose logging is added.

## Future Extension
After this architecture is stable, add optional manifest fields:
- `tags` (memory, cache, occupancy, atomics)
- `requires_timestamps`
- `requires_feature_bits`
- `default_enabled`

This allows future capability gating without changing main control flow.
