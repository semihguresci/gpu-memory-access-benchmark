# Generated Experiment Registry Architecture

## Purpose
This document describes the build-time generated experiment registry. The shared source of truth is [`config/experiment_manifest.json`](../../config/experiment_manifest.json), consumed by CMake and Python collection/report tooling.

The goal is to support many experiments without continuously editing control-flow code.

## Problem Summary
Before the generated registry, experiment dispatch was hardcoded:
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
    std::array<uint32_t, 3> required_local_size{1U, 1U, 1U};
    uint32_t required_shared_memory_bytes = 0;
};
```

## Build-Time Generation Model
The JSON manifest is read by `cmake/experiments_manifest.cmake` and `scripts/experiment_manifest.py`.

Minimum fields per experiment:
- `id` (example: `01_dispatch_basics`)
- `display_name`
- `category` (example: `foundations`, `memory_layout`)
- `adapter_symbol` (function symbol used in generated registry)
- `enabled` (boolean)
- `source` and `adapter_source` (repository-relative C++ source paths)
- `required_local_size` (three positive integers, largest local dimensions needed by the enabled variants)
- `required_shared_memory_bytes` (largest declared shared footprint among the enabled variant shaders)

Before invoking an adapter, the generic runtime checks the declared local dimensions, total invocation count, and shared-memory footprint against device limits. Update both requirements when adding or changing shaders. For example, Experiment 29 requires 33,792 shared bytes and Experiment 35 requires 32,768. Variant dispatch dimensions and subgroup-specific assumptions still need checks inside the implementation. `create_buffer_resource` separately rejects storage buffers larger than `maxStorageBufferRange`.

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

## Collection, analysis, and report refresh

Rebuild the selected executable after registration or shader changes, then collect and analyze the same run before refreshing reports:

```powershell
cmake --preset windows-tests-vs
cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments
python scripts/run_experiment_data_collection.py --experiment all --binary build-tests-vs/Release/gpu_memory_layout_experiments.exe --iterations 5 --warmup 2 --size 128M --label integrity_fixes
python scripts/generate_experiment_artifacts.py --experiment all
python scripts/refresh_experiment_reports.py --experiment all --check
python scripts/refresh_experiment_reports.py --experiment all
```

Use an explicit `--binary` if another checkout or build could be selected. The `128M` scratch example accommodates the radix-sort minimum; a whole-suite `4M` run fails clearly at that experiment. The 2026-10-23 refresh uses `32M` for individual experiments and `128M` for radix sort. These are exploratory sizes; larger working sets and repeated independent collections are needed for sustained-bandwidth claims. Each command also accepts one full experiment ID.

Current exports use schema `1.2.0` and record `measurement_revision=2026-11-08-complete-review-fixes` plus the actual command-line argument vector. The measurement revision identifies changed benchmark semantics; it is not a source, executable, or shader hash. Keep historical archives untouched and never relabel their rows with this revision. Preserve any additional collection hashes/provenance in the export and matching archive.

Analyzers select one measurement revision and reject failed, missing-correctness, and non-finite measurements. They write `results/tables/analysis_provenance.json` with the exact source JSON hashes, analyzer/helper hashes, and only CSV/chart paths actually written in that invocation. Separate plotters verify each requested CSV before reading it, then bind their newly generated PNG/SVG files and plotter hashes. Unused files left from earlier runs remain unbound, even if their modification time is recent. Changing a source, tool, or bound output requires regenerating analysis and plots.

The report refresher reads current raw JSON only, rejects failed/non-finite or incomplete rows, and recomputes medians at the largest size common to all variant/dispatch-count cases. If no common size exists, it labels each case's own largest size explicitly. It writes reports only after all selected exports pass validation, records runtime metadata and raw SHA-256, and links an archive only when its entire JSON content matches. Derived CSV/PNG/SVG links require verified manifest bindings and the exact current raw hash among the analysis sources. Missing or stale provenance omits derived links with a reason while retaining the raw-data report. The refresher does not run benchmarks/analyzers or change raw data, derived files, or archives.
