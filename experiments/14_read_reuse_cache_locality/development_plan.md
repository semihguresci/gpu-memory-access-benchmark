# Experiment 14 Read Reuse and Cache Locality: Codebase Development Plan

Date: 2026-03-27
Source spec: `docs/experiment_plans/14_read_reuse_cache_locality.md`

## 1. Scope and Current Status
This plan tracks implementation and workflow integration for Experiment 14 (`14_read_reuse_cache_locality`).

Current status:
- lecture-note experiment spec is present
- experiment-local README is present
- runtime architecture note is present
- runtime implementation, adapter wiring, shader, and analysis/plotting scripts are not present yet
- root workflow integration is not present yet in the manifest, build target list, or helper scripts
- no measured dataset or report is checked in under `experiments/14_read_reuse_cache_locality/results.md`

Primary implementation decisions for this experiment:
- implement Experiment 14 as deterministic pair-reuse index schedules over a fixed source buffer
- do not use a same-thread reread loop as the primary benchmark path, because compiler register reuse and common-subexpression elimination can blur cache-locality conclusions
- keep every unique source element touched exactly twice so only the second-touch distance changes across the main variants

## 2. Development Phases
### Phase A: Runtime Contract and Correctness
- [ ] Lock the primary variant set to `reuse_distance_1`, `reuse_distance_32`, `reuse_distance_256`, `reuse_distance_4096`, and `reuse_distance_full_span`.
- [ ] Keep logical invocation count fixed across variants and derive source capacity as `logical_count / 2`.
- [ ] Add deterministic pair-schedule generation where every source index appears exactly twice with controlled spacing.
- [ ] Keep the destination write sequential and identical for every variant so reuse distance remains the primary changing variable.
- [ ] Add deterministic CPU reference generation for the full destination span and validate the index buffer remains unchanged.
- [ ] Emit row-level output (`gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, `correctness_pass`, `notes`) with reuse metadata stored in `notes`.

### Phase B: Experiment Tooling
- [ ] Add experiment-local `collect_run.py`.
- [ ] Add experiment-local analysis and plotting scripts for reuse-distance slowdown, normalized speedup, and stability.
- [ ] Add results/report scaffolding (`results.md`, `results/charts/`, `results/tables/`, `runs/`, and script docs).

### Phase C: Root Workflow Integration
- [ ] Add `14_read_reuse_cache_locality` to `cmake/experiments_manifest.cmake`.
- [ ] Add Experiment 14 sources to `CMakeLists.txt`.
- [ ] Add `14_read_reuse_cache_locality` to `scripts/run_experiment_data_collection.py`.
- [ ] Add `14_read_reuse_cache_locality` to `scripts/generate_experiment_artifacts.py`.
- [ ] Verify shader auto-compilation resolves the Experiment 14 shader output with a unique basename.

### Phase D: Validation and Hardening
- [ ] Configure with `cmake --preset windows-tests-vs`.
- [ ] Build release target with `cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments`.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units when compile commands are available.
- [ ] Run tests and a benchmark smoke run for Experiment 14.

## 3. Acceptance Criteria Tracking
- [ ] `--experiment 14_read_reuse_cache_locality` is discoverable and runnable.
- [ ] Row-level output exists for the planned reuse-distance variants and includes source-span and reuse metadata.
- [ ] Adapter reports failure on correctness mismatch, odd-count contract violations, or non-finite timing.
- [ ] Root data collection and artifact-generation scripts support Experiment 14.
- [ ] Experiment-local `results.md` contains measured values, metadata, artifact links, and limitations.
