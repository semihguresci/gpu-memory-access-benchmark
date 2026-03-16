# Experiment 09 vec3 vs vec4 Padding Costs: Codebase Development Plan

Date: 2026-03-16  
Source spec: `docs/experiment_plans/09_vec3_vec4_padding_costs.md`

## 1. Scope and Current Status
This plan tracks implementation and workflow integration for Experiment 09 (`09_vec3_vec4_padding_costs`).

Current status:
- runtime experiment, shaders, adapter, and registry wiring are integrated
- deterministic correctness checks are integrated
- experiment-local run collection script is integrated
- root data-collection helper supports Experiment 09
- first measured dataset and result artifacts are generated

## 2. Development Phases
### Phase A: Runtime and Correctness
- [x] Implement `vec3_padded`, `vec4`, and `split_scalars` variants with equivalent update math.
- [x] Add explicit host layout contracts and static layout assertions.
- [x] Add deterministic seed/expected-value logic for correctness.
- [x] Emit per-iteration rows including `gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, and notes.
- [x] Enforce correctness fail behavior at adapter boundary.

### Phase B: Experiment Tooling
- [x] Add experiment-local `collect_run.py` script.
- [x] Create results folders and first summary artifacts (`.csv`, `.svg`).
- [x] Add initial `results.md` report with measured values and limits.

### Phase C: Root Workflow Integration
- [x] Add `09_vec3_vec4_padding_costs` to `scripts/run_experiment_data_collection.py`.
- [ ] Add `09_vec3_vec4_padding_costs` to `scripts/generate_experiment_artifacts.py`.

### Phase D: Validation and Hardening
- [x] Configure with `windows-tests-vs` preset.
- [x] Build Debug target.
- [x] Run `clang-format` on touched C++ files.
- [x] Run `clang-tidy` for modified translation units.
- [x] Run test suite (`ctest`, Debug profile).

## 3. Acceptance Criteria Tracking
- [x] `--experiment 09_vec3_vec4_padding_costs` is discoverable and runnable.
- [x] Row-level output exists for all three variants.
- [x] Adapter reports failure on correctness mismatch.
- [x] Data collection script can run and collect Experiment 09 output.
- [x] `results.md` includes concrete measured values, artifact links, and limitations.
