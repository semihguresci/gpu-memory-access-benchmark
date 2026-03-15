# Experiment 07 AoSoA Blocked Layout: Codebase Development Plan

Date: 2026-03-15  
Source spec: `docs/experiment_plans/07_aosoa_blocked_layout.md`

## 1. Scope and Current Status
This plan tracks implementation and analysis workflow setup for Experiment 07 (`07_aosoa_blocked_layout`).

Current status:
- runtime experiment, shader wiring, adapter, and registry entry are integrated
- row-level output and deterministic correctness checks are integrated
- experiment-local scripts are present (`collect`, `analyze`, `plot`)
- first measured dataset is still pending

## 2. Development Phases
### Phase A: Runtime and Correctness
- [x] Implement AoS and SoA baseline variants for this experiment.
- [x] Implement blocked AoSoA variant with block-size sweep (`4, 8, 16, 32`).
- [x] Add deterministic seed/expected-value logic for correctness.
- [x] Add per-iteration row export with `gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, and notes.
- [x] Enforce correctness pass/fail at adapter boundary.

### Phase B: Experiment Tooling
- [x] Add experiment-local run collection script.
- [x] Add experiment-local multi-run analysis script.
- [x] Add quick single-run plotting script.
- [x] Add results/report scaffolding and output directories.

### Phase C: Root Workflow Integration
- [x] Add `07_aosoa_blocked_layout` support to data collection helper script.
- [x] Add `07_aosoa_blocked_layout` support to artifact generation helper script.
- [x] Produce first benchmark JSON and collect into `runs/` (smoke run).

### Phase D: Validation and Hardening
- [x] Run warning-clean build for modified sources.
- [x] Run `clang-format` on touched C++ files.
- [x] Run `clang-tidy` for touched translation units when compile database is available.
- [x] Run unit tests and targeted regression checks.

## 3. Acceptance Criteria Tracking
- [x] `--experiment 07_aosoa_blocked_layout` is discoverable and runnable from CLI.
- [x] Row-level output exists for `aos`, `soa`, and `aosoa_b{4,8,16,32}` variants.
- [x] Adapter fails run on correctness mismatch.
- [x] Experiment-local scripts and result directories are present.
- [x] Reproducible dataset and charts are generated under `experiments/07_aosoa_blocked_layout/results/` (smoke quality).
- [x] `results.md` is updated with measured values and limitations.
