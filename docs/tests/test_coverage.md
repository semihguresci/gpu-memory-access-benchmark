# Test Coverage

## Coverage Summary
This document maps current automated tests to production code and highlights known gaps.

Current automated suite:
- targets:
  - `gpu_memory_layout_unit_tests` (default)
  - `gpu_memory_layout_integration_tests` (optional)
- framework: GoogleTest
- unit test files:
  - `tests/unit/app_options_tests.cpp`
  - `tests/unit/benchmark_runner_tests.cpp`
  - `tests/unit/json_exporter_tests.cpp`
  - `tests/unit/vulkan_compute_utils_tests.cpp`
- integration test files:
  - `tests/integration/vulkan_context_integration_tests.cpp` (built only when `ENABLE_GPU_INTEGRATION_TESTS=ON`)
- shared helpers:
  - `tests/support/app_options_test_utils.hpp`
  - `tests/support/filesystem_test_utils.hpp`

## Feature-to-Test Mapping

### `src/utils/app_options.cpp`
Covered behaviors:
- parses `all` experiment selection
- parses comma-separated selections with trimming
- deduplicates repeated experiment IDs
- normalizes output path to `.json`
- exits with code `2` on unknown experiment ID
- exits with code `2` on invalid `--size`

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
- writes metadata with correct count fields
- writes results array
- writes optional rows array when provided
- creates nested output directory path

Covered by:
- `JsonExporterTests.*` in `tests/unit/json_exporter_tests.cpp`

### `src/utils/vulkan_compute_utils.cpp` (CPU-only subset)
Covered behaviors:
- `compute_group_count_1d` rounding and zero-guard behavior
- `resolve_shader_path` explicit-path and fallback resolution behavior
- `read_binary_file` successful read and empty-file failure behavior

Covered by:
- `VulkanComputeUtilsTests.*` in `tests/unit/vulkan_compute_utils_tests.cpp`

## Out-of-Scope / Not Yet Covered

### GPU/Vulkan runtime paths
Partially covered by optional integration tests:
- `src/vulkan_context.cpp` basic init/shutdown smoke path

Still not covered:
- `src/utils/buffer_utils.cpp`
- `src/utils/gpu_timestamp_timer.cpp`
- Vulkan-object creation and destruction success/failure paths
- synchronization barrier command recording correctness

Risk:
- runtime regressions may only appear on actual GPU execution

### Experiment implementations
Not currently covered by automated tests:
- `src/experiments/dispatch_basics_experiment.cpp`
- `src/experiments/aos_soa_experiment.cpp`
- adapter files in `src/experiments/adapters/`
- generated registry integration in end-to-end run

Risk:
- shader path regressions, descriptor/pipeline setup failures, and correctness-path issues can slip through

### Entrypoint and orchestration
Not currently covered by automated tests:
- `src/main.cpp` orchestration and failure handling paths

## Coverage Quality Notes
- current tests are strong for deterministic utility logic and serialization contracts
- current tests are intentionally lightweight and platform-friendly for fast local development
- CTest labels allow focused execution (`unit`, `integration`)
- GPU integration remains a gated stage for capable environments

## Next Coverage Targets
1. Add integration tests that run one minimal adapter case when GPU timestamps are available.
2. Add failure-path tests for missing shader binaries in experiment adapters.
3. Add contract test for experiment registry consistency (`enabled_experiment_ids`, descriptor lookup).
