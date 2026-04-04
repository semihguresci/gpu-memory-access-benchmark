# Experiment 13 Scatter Access Pattern: Codebase Development Plan

Date: 2026-03-26
Source spec: `docs/experiment_plans/13_scatter_access_pattern.md`

## 1. Scope and Current Status
This plan tracks implementation and workflow integration for Experiment 13 (`13_scatter_access_pattern`).

Current status:
- lecture-note experiment spec is present
- experiment-local README is present
- runtime architecture note is not present yet
- runtime implementation, adapter wiring, shader, and analysis/plotting scripts are not present yet
- root workflow integration is not present yet in the manifest, build target list, or helper scripts
- no measured dataset or report is checked in under `experiments/13_scatter_access_pattern/results.md`

Primary implementation decision for this experiment:
- keep the primary comparison on one deterministic atomic scatter kernel (`atomicAdd(dst[target], 1u)`)
- do not use non-atomic colliding variants in the main result set, because that would trade a performance experiment for undefined race behavior

## 2. Development Phases
### Phase A: Runtime Contract and Correctness
- [ ] Lock the primary distribution set to a no-collision baseline plus low-collision and high-collision scatter cases.
- [ ] Add deterministic target generation with a fixed seed and exact contention shaping.
- [ ] Keep logical work fixed across variants and keep destination capacity constant so distribution is the main changing variable.
- [ ] Add deterministic CPU reference histogram generation for the full destination span.
- [ ] Validate every destination counter after dispatch and fail the run on any mismatch.
- [ ] Emit row-level output (`gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, `correctness_pass`, `notes`) with contention metadata stored in `notes`.

### Phase B: Experiment Tooling
- [ ] Add experiment-local `collect_run.py`.
- [ ] Add experiment-local analysis and plotting scripts for slowdown, throughput, and stability.
- [ ] Add results/report scaffolding (`results.md`, `results/charts/`, `results/tables/`, `runs/`, and script docs).

### Phase C: Root Workflow Integration
- [ ] Add `13_scatter_access_pattern` to `cmake/experiments_manifest.cmake`.
- [ ] Add Experiment 13 sources to `CMakeLists.txt`.
- [ ] Add `13_scatter_access_pattern` to `scripts/run_experiment_data_collection.py`.
- [ ] Add `13_scatter_access_pattern` to `scripts/generate_experiment_artifacts.py`.
- [ ] Verify shader auto-compilation resolves the Experiment 13 shader output with a unique basename.

### Phase D: Validation and Hardening
- [ ] Configure with `cmake --preset windows-tests-vs`.
- [ ] Build release target with `cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments`.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units when compile commands are available.
- [ ] Run tests and a benchmark smoke run for Experiment 13.

## 3. Acceptance Criteria Tracking
- [ ] `--experiment 13_scatter_access_pattern` is discoverable and runnable.
- [ ] Row-level output exists for the planned scatter variants and includes contention metadata.
- [ ] Adapter reports failure on correctness mismatch or non-finite timing.
- [ ] Root data collection and artifact-generation scripts support Experiment 13.
- [ ] Experiment-local `results.md` contains measured values, metadata, artifact links, and limitations.
