# Test Plan

## Scope
This plan defines how to prevent regressions while developing Vulkan compute benchmark features.

Primary goals:
- catch behavioral regressions early in utility and orchestration code
- keep tests deterministic and fast for day-to-day development
- isolate GPU-dependent checks from CPU-only unit tests

## Test Levels

### 1. Unit Tests (fast, deterministic, no live GPU required)
Current target: `gpu_memory_layout_unit_tests`

Focus areas:
- CLI parsing and validation (`ArgumentParser`)
- benchmark statistics and iteration behavior (`BenchmarkRunner`)
- JSON schema/output generation (`JsonExporter`)
- pure utility behavior (`VulkanComputeUtils` helpers that do not require live Vulkan handles)

Execution expectation:
- runs locally in under a few seconds
- no external services required

### 2. Integration Tests (GPU/runtime dependent)
Scope:
- initialize `VulkanContext`
- run selected experiment adapters end-to-end
- validate output file presence and minimum schema correctness

Status:
- optional target is available: `gpu_memory_layout_integration_tests`
- currently includes `VulkanContext` smoke initialization test
- disabled by default and enabled with `-DENABLE_GPU_INTEGRATION_TESTS=ON`

### 3. Regression Tests
Regression tests are kept close to bug fixes.

Policy:
- each production bug fix should include at least one test that failed before the fix
- regression tests should prefer smallest reproducible surface first (unit-level if possible)
- keep historical failure mode in test name or test comments when needed

## Feature Test Strategy

### App options and CLI
- verify accepted argument formats (`--size`, `--experiment`)
- verify error exits for invalid inputs
- verify normalization behavior (`.json` output suffix)

### Benchmark metrics
- verify finite-sample filtering
- verify summary math (`average`, `median`, `p95`, min/max)
- verify warmup and timed iteration counts are honored

### JSON export contract
- verify schema version/name are emitted
- verify metadata and row/result count consistency
- verify optional `rows` field behavior

### Compute utility helpers
- verify group-count rounding rules
- verify shader path resolution precedence
- verify binary file read success/failure behavior

## Pass/Fail Gates

For regular development:
1. Configure with tests enabled.
2. Build `gpu_memory_layout_tests` (aggregates unit + optional integration).
3. Run `ctest`.
4. Merge only when tests pass.

Recommended commands:

```powershell
cmake -S . -B build-tests-vs -G "Visual Studio 18 2026" -A x64 -DBUILD_TESTING=ON -DBUILD_SHADERS=OFF
cmake --build build-tests-vs --config Debug --target gpu_memory_layout_tests
ctest --test-dir build-tests-vs -C Debug -L unit --output-on-failure
```

Optional GPU integration run:

```powershell
cmake -S . -B build-tests-vs -G "Visual Studio 18 2026" -A x64 -DBUILD_TESTING=ON -DBUILD_SHADERS=OFF -DENABLE_GPU_INTEGRATION_TESTS=ON
cmake --build build-tests-vs --config Debug --target gpu_memory_layout_integration_tests
ctest --test-dir build-tests-vs -C Debug -L integration --output-on-failure
```

## Test Data and Environment Rules
- use temporary directories for filesystem tests
- avoid dependency on repository-local mutable files
- do not rely on system GPU state in unit tests

## Expansion Roadmap
1. Add adapter-level integration tests for `01_dispatch_basics` and `06_aos_soa`.
2. Add schema compatibility tests for future JSON schema version bumps.
3. Add negative tests for shader/module load failures in adapter paths.
4. Add contract tests for generated experiment registry behavior.
