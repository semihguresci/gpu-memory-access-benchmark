# Experiment 12 Gather Access Pattern: Codebase Development Plan

Date: 2026-03-25
Source spec: `docs/experiment_plans/12_gather_access_pattern.md`

## 1. Scope and Current Status
This plan tracks implementation and workflow integration for Experiment 12 (`12_gather_access_pattern`).

Current status:
- lecture-note experiment spec is present
- experiment-local README is present
- runtime architecture note is present
- runtime implementation, adapter wiring, shader, and analysis/plotting scripts are present
- no measured dataset or report is checked in under `experiments/12_gather_access_pattern/results.md` yet

## 2. Development Phases
### Phase A: Runtime Contract and Correctness
- [x] Decide the exact gather distribution set and seed policy.
- [x] Define the source, index, and destination buffer sizes so the logical work stays fixed across distributions.
- [x] Add deterministic CPU reference generation and correctness checks for the gathered destination values.
- [x] Define how any padding or alignment slack is handled during validation.
- [x] Emit row-level output (`gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, `correctness_pass`, notes).

### Phase B: Experiment Tooling
- [x] Add experiment-local `collect_run.py` script.
- [x] Add experiment-local analysis/plotting script for distribution versus throughput trends.
- [ ] Add results/report scaffolding (`results.md` and script docs).

### Phase C: Root Workflow Integration
- [x] Add `12_gather_access_pattern` to `cmake/experiments_manifest.cmake`.
- [x] Add `12_gather_access_pattern` to `scripts/run_experiment_data_collection.py`.
- [x] Add `12_gather_access_pattern` to `scripts/generate_experiment_artifacts.py`.
- [x] Verify shader auto-compilation resolves experiment 12 shader outputs with a unique basename.

### Phase D: Validation and Hardening
- [x] Configure with `cmake --preset windows-tests-vs`.
- [x] Build release target with `cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments`.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units when compile commands are available.
- [x] Run tests and a benchmark smoke run.

## 3. Acceptance Criteria Tracking
- [x] `--experiment 12_gather_access_pattern` is discoverable and runnable.
- [x] Row-level output exists for the planned distributions and includes logical size metadata.
- [x] Adapter reports failure on correctness mismatch.
- [x] Root data collection and artifact-generation scripts support Experiment 12.
- [ ] Experiment-local `results.md` contains measured values, metadata, artifact links, and limitations.
