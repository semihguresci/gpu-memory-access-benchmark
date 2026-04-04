# Experiment 16 Shared Memory Tiling: Codebase Development Plan

Date: 2026-03-27
Source spec: `docs/experiment_plans/16_shared_memory_tiling.md`

## 1. Scope and Current Status
This plan tracks implementation and workflow integration for Experiment 16 (`16_shared_memory_tiling`).

Current status:
- lecture-note experiment spec is present
- experiment-local README is present
- runtime architecture note is not present yet
- runtime implementation, adapter wiring, shaders, and analysis/plotting scripts are not present yet
- root workflow integration is not present yet in the manifest, build target list, or helper scripts
- no measured dataset or report is checked in under `experiments/16_shared_memory_tiling/results.md`

Primary implementation decisions for this experiment:
- use a 1D sliding-window sum over a padded source buffer so boundary handling does not dominate the measurement
- keep `local_size_x` fixed at one default value in Experiment 16 and reserve tile-size tuning for Experiment 17
- compare two explicit shader paths (`direct_global` and `shared_tiled`) instead of using a runtime branch inside one shader
- sweep reuse radius (`1`, `4`, `8`, `16`) so the break-even point between reuse benefit and synchronization cost becomes visible

## 2. Development Phases
### Phase A: Runtime Contract and Correctness
- [ ] Lock the radius sweep and padded buffer layout so all cases share one logical output span.
- [ ] Derive the logical output count from the existing scratch-size budget while fitting the max-radius padded source buffer and output buffer.
- [ ] Add deterministic bounded `uint32_t` source generation so exact CPU validation stays safe and reproducible.
- [ ] Add CPU reference generation for every output element at every radius.
- [ ] Emit row-level output (`gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, `correctness_pass`, `notes`) with tiling metadata stored in `notes`.
- [ ] Fail the run on any mismatch or non-finite timing sample.

### Phase B: Runtime and Shader Paths
- [ ] Add experiment-local config/output contract and adapter.
- [ ] Add separate direct and tiled compute shaders with unique basenames under `shaders/16_shared_memory_tiling/`.
- [ ] Add explicit buffer, descriptor, and pipeline setup/teardown for both variants.
- [ ] Validate timestamp support and shader resolution before running measurements.
- [ ] Preserve reverse-order teardown and handle reset rules for every Vulkan object used by the experiment.

### Phase C: Experiment Tooling
- [ ] Add experiment-local `collect_run.py`.
- [ ] Add experiment-local analysis and plotting scripts for direct-versus-tiled speedup by reuse radius.
- [ ] Add results/report scaffolding (`results.md`, `results/charts/`, `results/tables/`, `runs/`, and script docs).

### Phase D: Root Workflow Integration
- [ ] Add `16_shared_memory_tiling` to `cmake/experiments_manifest.cmake`.
- [ ] Add Experiment 16 sources to `CMakeLists.txt`.
- [ ] Add `16_shared_memory_tiling` to `scripts/run_experiment_data_collection.py`.
- [ ] Add `16_shared_memory_tiling` to `scripts/generate_experiment_artifacts.py`.
- [ ] Verify recursive shader auto-compilation resolves the Experiment 16 shader outputs with unique basenames.

### Phase E: Validation and Hardening
- [ ] Configure with `cmake --preset windows-tests-vs`.
- [ ] Build release target with `cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments`.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units when compile commands are available.
- [ ] Run tests and a benchmark smoke run for Experiment 16.

## 3. Acceptance Criteria Tracking
- [ ] `--experiment 16_shared_memory_tiling` is discoverable and runnable.
- [ ] Row-level output exists for the planned `variant x radius` cases and includes tiling metadata in `notes`.
- [ ] Adapter reports failure on correctness mismatch or non-finite timing.
- [ ] Root data collection and artifact-generation scripts support Experiment 16.
- [ ] Experiment-local `results.md` contains measured values, metadata, artifact links, and limitations.
- [ ] Experiment 17 can reuse the tiled kernel contract without changing the Experiment 16 correctness model.
