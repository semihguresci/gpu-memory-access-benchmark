# Experiment 02 Local Size Sweep: Codebase Development Plan

Date: 2026-03-14  
Source spec: `docs/experiment_plans/02_local_size_sweep.md`

## 1. Scope and Current Status
This plan tracks implementation of canonical Experiment 02 in the current Vulkan benchmark harness.

Current status:
- runtime implementation for `02_local_size_sweep` is not yet integrated
- no Experiment 02 adapter is registered in `cmake/experiments_manifest.cmake`
- experiment-local architecture/development/implementation planning docs are now present

## 2. Development Phases
### Phase A: Design Freeze and Constraints
- [ ] Confirm sweep matrix (`local_size_x`, `problem_size`, `dispatch_count`) and default values.
- [ ] Decide shader strategy (multiple fixed-local-size shaders vs specialization constants).
- [ ] Lock legality filter behavior and skip/fail policy.
- [ ] Align result-encoding strategy (`variant` naming and `notes` tags).

Phase completion gate:
- [ ] Experiment-plan decisions are stable enough to start coding without interface churn.

### Phase B: Shader and Pipeline Preparation
- [ ] Add Experiment 02 shader assets under `shaders/` with deterministic kernel behavior.
- [ ] Validate shaders compile via existing CMake shader build flow.
- [ ] Ensure one-to-one mapping from candidate local size to shader artifact.

Phase completion gate:
- [ ] All targeted local-size shader binaries are produced in build output.

### Phase C: Core Runtime Implementation
- [ ] Add `LocalSizeSweepExperiment` public API in `include/experiments/`.
- [ ] Implement `src/experiments/local_size_sweep_experiment.cpp`.
- [ ] Implement explicit Vulkan resource ownership and teardown in reverse creation order.
- [ ] Integrate deterministic correctness checks for both primary and control variants.
- [ ] Capture row-level metrics compatible with current JSON exporter schema.

Phase completion gate:
- [ ] Experiment function returns non-empty summaries and rows on at least one test GPU.

### Phase D: Adapter and Registry Integration
- [ ] Add `src/experiments/adapters/local_size_sweep_adapter.cpp`.
- [ ] Register experiment in `cmake/experiments_manifest.cmake` with canonical ID `02_local_size_sweep`.
- [ ] Reconfigure build and verify generated registry exposes the new experiment ID.
- [ ] Confirm `--experiment 02_local_size_sweep` CLI path works end-to-end.

Phase completion gate:
- [ ] Main runtime can execute Experiment 02 without manual dispatch code changes.

### Phase E: Analysis and Artifact Pipeline
- [ ] Add experiment-local scripts under `experiments/02_local_size_sweep/scripts/`.
- [ ] Generate pivot tables and primary charts under `results/tables/` and `results/charts/`.
- [ ] Add `results.md` summarizing measured outcomes and recommendation.

Phase completion gate:
- [ ] One reproducible run can produce a full minimum artifact set.

### Phase F: Validation and Hardening
- [ ] Run warning-clean builds (MSVC `/W4`, Clang/GCC warning set).
- [ ] Run `clang-format` and `clang-tidy` on touched C++ files.
- [ ] Execute unit tests and relevant integration checks.
- [ ] Perform at least one validation-enabled smoke run of Experiment 02.

Phase completion gate:
- [ ] No blocking correctness/resource-lifecycle issues remain.

## 3. Acceptance Criteria Tracking
- [ ] `--experiment 02_local_size_sweep` is discoverable and runnable from CLI.
- [ ] Illegal local sizes are filtered safely with explicit diagnostics.
- [ ] Every measured point emits correctness and timing outputs.
- [ ] Result export includes median/p95 summaries for each measured point.
- [ ] Analysis artifacts identify and justify a recommended local size.

## 4. Dependencies and Sequencing
- Phase A must complete before B and C.
- Phase B must complete before full C validation (runtime requires compiled shaders).
- Phase C must complete before D.
- Phase D must complete before E and F end-to-end verification.

Critical path:
- `shader assets -> experiment runtime -> adapter/registry -> run outputs -> analysis artifacts`

## 5. Risk Register
- Risk: local-size candidates exceed device limits on some GPUs.
  Mitigation: runtime legality filtering and skip reporting.
- Risk: per-size shader compilation differences confound conclusions.
  Mitigation: keep kernel body identical and only vary local-size declaration.
- Risk: run time becomes too long with full matrix.
  Mitigation: allow reduced problem-size subset for smoke runs; keep full sweep for report runs.
- Risk: correctness regressions hidden by timing-only focus.
  Mitigation: enforce row-level correctness gate before performance interpretation.

## 6. Definition of Done Status
- [ ] Core runtime module implemented and integrated.
- [ ] Experiment registry includes `02_local_size_sweep`.
- [ ] One complete dataset generated with charts/tables.
- [ ] Recommendation documented with explicit limitations and reproducibility command.
