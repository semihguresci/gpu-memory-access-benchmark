# Experiment 06 AoS vs SoA: Codebase Development Plan

Date: 2026-03-15  
Source spec: `docs/experiment_plans/06_aos_vs_soa.md`

## 1. Scope and Current Status
This plan tracks implementation and analysis workflow setup for Experiment 06 (`06_aos_vs_soa`).

Current status:
- runtime experiment, shader wiring, adapter, and registry entry are integrated
- row-level output and deterministic correctness checks are integrated
- experiment-local scripts are present (`collect`, `analyze`, `plot`)
- first measured dataset is still pending

## 2. Development Phases
### Phase A: Runtime and Correctness
- [x] Verify AoS host struct layout against shader expectations.
- [x] Implement deterministic seed/expected-value logic for correctness.
- [x] Add per-iteration row export with `gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, and notes.
- [x] Enforce correctness pass/fail at adapter boundary.

### Phase B: Experiment Tooling
- [x] Add experiment-local run collection script.
- [x] Add experiment-local multi-run analysis script.
- [x] Add quick single-run plotting script.
- [x] Add results/report scaffolding and output directories.

### Phase C: Root Workflow Integration
- [x] Add `06_aos_vs_soa` support to data collection helper script.
- [x] Add `06_aos_vs_soa` support to artifact generation helper script.
- [ ] Produce first benchmark JSON and collect into `runs/`.

### Phase D: Validation and Hardening
- [ ] Run warning-clean build for modified sources.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for touched translation units when compile commands are available.
- [ ] Run unit tests and targeted regression checks.

## 3. Acceptance Criteria Tracking
- [x] `--experiment 06_aos_vs_soa` is discoverable and runnable from CLI.
- [x] Row-level output exists for both `aos` and `soa` variants.
- [x] Adapter fails run on correctness mismatch.
- [x] Experiment-local scripts and result directories are present.
- [ ] Reproducible dataset and charts are generated under `experiments/06_aos_vs_soa/results/`.
- [ ] `results.md` is updated with measured values and limitations.
