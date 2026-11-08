# Test Coverage

## Coverage Summary
This document maps current automated tests to production code and highlights known gaps.

Current automated suite:
- CTest discovers 57 GoogleTest cases: 50 unit cases and seven integration cases when GPU integration is enabled.
- The representative GPU matrix is one integration case containing 26 experiment invocations; it is not 26 separate CTest cases.
- Python reporting, analysis, workflow, and provenance suites contain 74 tests, run separately from CTest.
- targets:
  - `gpu_memory_layout_unit_tests` (default)
  - `gpu_memory_layout_integration_tests` (optional)
- framework: GoogleTest
- unit test files:
  - `tests/unit/app_options_tests.cpp`
  - `tests/unit/benchmark_runner_tests.cpp`
  - `tests/unit/experiment_metrics_tests.cpp`
  - `tests/unit/float_validation_tests.cpp`
  - `tests/unit/gpu_timestamp_timer_tests.cpp`
  - `tests/unit/json_exporter_tests.cpp`
  - `tests/unit/scalar_type_width_utils_tests.cpp`
  - `tests/unit/scratch_buffer_budget_tests.cpp`
  - `tests/unit/vulkan_compute_utils_tests.cpp`
- integration test files:
  - `tests/integration/vulkan_context_integration_tests.cpp`
  - `tests/integration/experiment_smoke_tests.cpp`
  - `tests/integration/output_freshness_tests.cpp`
- integration tests are built only when `ENABLE_GPU_INTEGRATION_TESTS=ON`; GPU-dependent cases skip when Vulkan initialization or timestamp support is unavailable.
- shared helpers:
  - `tests/support/app_options_test_utils.hpp`
  - `tests/support/filesystem_test_utils.hpp`

## Feature-to-Test Mapping

### `src/utils/app_options.cpp`
Covered behaviors:
- parses `all` experiment selection
- parses comma-separated selections with trimming
- deduplicates repeated experiment IDs
- parses the verbose-progress flag
- normalizes output path to `.json`
- exits with code `2` on unknown experiment ID
- exits with code `2` on invalid `--size`
- rejects signed and overflowing sizes while accepting the largest representable byte/suffix values without allocation

Covered by:
- `AppOptionsTests.*` in `tests/unit/app_options_tests.cpp`

### `src/benchmark_runner.cpp`
Covered behaviors:
- computes summary metrics (`average`, `min`, `max`, `median`, `p95`)
- filters non-finite samples
- returns `NaN` metrics when no finite samples exist
- respects warmup/timed iteration counts in `run_timed`

Covered by:
- `BenchmarkRunnerTests.*` in `tests/unit/benchmark_runner_tests.cpp`

### `src/utils/json_exporter.cpp`
Covered behaviors:
- emits expected schema name/version
- writes metadata with correct count fields and six-digit fractional UTC export timestamps
- writes results array
- writes optional rows array when provided
- creates nested output directory path
- rejects an existing directory as the output path rather than reporting a successful write

Covered by:
- `JsonExporterTests.*` in `tests/unit/json_exporter_tests.cpp`

### `scripts/refresh_experiment_reports.py`

Covered by `tests/scripts/test_refresh_experiment_reports.py` using Python's standard-library `unittest`:

- rejects another experiment's rows, failed correctness, non-finite numbers, empty exports, and missing timed iterations;
- selects the largest common problem size and keeps different dispatch counts separate, with a labeled fallback when no common size exists;
- computes report medians from raw rows and preserves measurement revision and collection provenance;
- links archives only when the full JSON content matches the current export;
- links derived CSV/PNG/SVG outputs only when explicit provenance binds their hashes and analyzer/plotter versions to sources including the exact current raw export;
- omits stale or unbound outputs, including obsolete leftovers with recent modification times, while preserving raw-data summaries.

Run independently of CTest:

```powershell
python -m unittest discover -s tests/scripts -p test_refresh_experiment_reports.py -v
python -m unittest discover -s scripts/tests -v
python scripts/refresh_experiment_reports.py --experiment all --check
```

The final command validates and renders reports without writing them. After fresh collection and artifact generation, omit `--check` to replace each selected `results.md`. These checks verify reporting integrity; they do not replace GPU correctness runs or statistical performance analysis. Current raw exports identify the corrected semantics with `measurement_revision=2026-11-08-remaining-review-fixes`, with experiments 38/40 refreshed under `2026-11-08-contention-merge` after integrating the contention optimizations. The revision is not a source/binary hash.

The `scripts/tests` suites cover measurement-revision selection, exact-current source preference, fail-closed row validation, separate fp32 precision errors, requested-table binding before plotting, and source/tool/table/chart tampering. Explicit output lists prevent omitted optional tables and renamed charts from being relabeled as current artifacts. Historical archive analysis remains available within its own revision and is identified separately from the current raw report.

The analysis regressions also separate validation/warmup/timed configurations throughout experiment 01 comparisons and preserve experiment 22 traffic-model metadata. Same-second legacy archives use numeric collector sequences, while fractional export timestamps take precedence over archive names and copy times. Archive-only generation for experiments 46–48 invokes both analysis and plotting.

The workflow regressions also cover external destinations for every collector, export-timestamp ordering in every single-archive analyzer, modification-time fallback, artifact-only GUI execution without a binary or valid benchmark settings, exact nonpower size labels, and automatic CMake regeneration after manifest-only changes.

### `src/utils/vulkan_compute_utils.cpp` (CPU-only subset)
Covered behaviors:
- `compute_group_count_1d` rounding and zero-guard behavior
- large-count ceiling arithmetic without uint32 intermediate overflow
- device-limit validation for per-axis workgroup size, total invocations, shared-memory footprint, zero-sized workgroups, and both storage-buffer descriptor count limits
- `resolve_shader_path` explicit-path and fallback resolution behavior
- `read_binary_file` successful read and empty-file failure behavior

Covered by:
- `VulkanComputeUtilsTests.*` in `tests/unit/vulkan_compute_utils_tests.cpp`

### Timestamp, metrics, scratch-budget, and scalar-width helpers

CPU unit coverage includes:

- `GpuTimestampTimerTests.*`: timestamp counter masking/wraparound, full-width 64-bit counters, zero elapsed ticks, and invalid counter widths. These test arithmetic independently of a physical GPU's timestamp width.
- `ExperimentMetricsTests.*`: invalid timing/input handling and payload/dispatch accounting for throughput and GB/s.
- `ScratchBufferBudgetTests.*`: per-buffer division, scaled budgets without intermediate overflow, and invalid/zero inputs.
- `ScalarTypeWidthUtilsTests.*`: packed storage sizing including uint32 limits, deterministic bounded seeds, clamping/update rules, half/narrow conversion, packed-lane preservation, and representation-specific validation tolerances.
- `FloatValidationTests.*`: finite absolute/relative tolerance behavior and rejection of infinities, NaNs, and invalid tolerances, shared by the layout experiments.

### Runtime, experiments, and generated-registry integration

The integration target checks benchmark-binary availability, Vulkan context initialization/shutdown, and timestamp availability. Its 26-case matrix invokes the real executable with `--iterations 1 --warmup 0`, exercising CLI parsing, generated-registry lookup, adapters, resource/pipeline setup, dispatch, validation, teardown, and JSON export on successful paths.

The matrix covers experiment IDs **01, 03, 06, 11, 15, 16, 17, 18, 20, 22, 24, 27, 28, 32, 34, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, and 48**. It checks process success, nonempty rows/results, matching experiment IDs, metadata counts/configuration, and every exported row's correctness. Export-contract assertions verify source-request byte accounting in 22, six-buffer budgets in 28/47, survivor traffic in 32, disjoint frame outputs in 37, workgroup aggregation and atomic request accounting in 38/40, visibility-sensitive traffic in 44, and active versus resident payload in 46. Experiments 03/15 exercise the 48-byte valid boundary; separate tests reject smaller probe budgets and budgets unable to hold experiment 47's eviction buffer.

The fresh-output integration regression runs a correct occupancy variant before a substituted no-op variant, verifies that the no-op measurements fail, and verifies that the remaining correct variant still passes.

The separate all-48 experiment validation/collection workflow provides broader run evidence; it does not expand the automated 26-case regression matrix or establish performance stability. Smoke cases use small configurations and do not enable validation layers themselves.

## Partial Coverage and Remaining Gaps

### GPU/Vulkan runtime paths

Successful buffer allocation/mapping, timestamp-query measurement, synchronization, and resource teardown execute through the representative experiment runs. Timestamp delta arithmetic and compute-limit rejection also have CPU unit coverage.

Remaining gaps include:

- injected Vulkan allocation, mapping, submission, and query failures, and exhaustive leak/cleanup checks;
- direct failure-path coverage for storage-descriptor limits and unavailable memory types;
- synchronization correctness under other legal subgroup mappings, queue capabilities, devices, and drivers;
- validation-layer synchronization diagnostics as an automated assertion, rather than inference from correct readback;
- physical timestamp-wrap/timeout behavior on devices with narrow counters.

### Experiment implementations

Dispatch Basics, AoS/SoA, and the other listed experiments and adapters have end-to-end smoke coverage. Experiments outside the 26-case matrix do not yet have automated GPU regression invocations. Existing cases do not exhaust every tail size, variant/device-limit combination, algorithmic distribution, or multi-iteration dependency.

Missing shader files and shader/pipeline creation failures still need dedicated adapter-level failure tests. Compiler register allocation, spills, cache transactions, and physical DRAM traffic require additional tooling beyond correctness tests.

### Entrypoint and orchestration

Successful single-experiment orchestration in `src/main.cpp`, CLI experiment resolution, and generated-registry dispatch are exercised by the matrix. CLI rejection cases have unit tests. Full `--experiment all` orchestration, manifest-generation failure fixtures, and export/submission failure propagation through the executable need dedicated regression coverage.

## Coverage Quality Notes
- current tests are strong for deterministic utility logic and serialization contracts
- current tests are intentionally lightweight and platform-friendly for fast local development
- CTest labels allow focused execution (`unit`, `integration`)
- GPU integration remains a gated stage for capable environments

## Next Coverage Targets
1. Add missing-shader and injected Vulkan/IO failure cases that check actionable errors and cleanup.
2. Add a validation-layer synchronization regression mode and targeted tail/limit/subgroup-mapping cases.
3. Add manifest-generation fixtures for malformed requirements/duplicate IDs and executable-level failure-propagation checks.
4. Expand GPU regression coverage beyond the current 26 experiments while retaining bounded test runtime.
