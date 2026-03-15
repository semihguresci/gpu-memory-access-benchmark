# Experiment 04 Sequential Indexing: Codebase Development Plan

Date: 2026-03-15  
Source spec: `docs/experiment_plans/04_sequential_indexing.md`

## 1. Scope and Current Status
This plan tracks implementation of canonical Experiment 04 in the Vulkan benchmark harness.

Current status:
- runtime implementation for `04_sequential_indexing` is integrated
- experiment is registered in `cmake/experiments_manifest.cmake`
- shader and adapter wiring are present
- analysis script scaffolding is present, but measured data is pending

## 2. Development Phases
### Phase A: Design Freeze and Constraints
- [x] Lock mapping policy: invocation `i` reads/writes element `i`.
- [x] Lock fixed local size and runtime clamp behavior.
- [x] Lock sweep range policy and dispatch-count sweep derived from Experiment 01.
- [x] Lock correctness validation strategy.

### Phase B: Shader and Pipeline Preparation
- [x] Add dedicated sequential indexing shader.
- [x] Validate naming convention and runtime lookup path.
- [ ] Validate shader binaries across all configured build presets.

### Phase C: Core Runtime Implementation
- [x] Add experiment API and implementation in `include/experiments/` and `src/experiments/`.
- [x] Implement explicit Vulkan ownership and reverse-order teardown.
- [x] Add warmup/timed execution loops with row-level output.
- [x] Add deterministic correctness checks against CPU reference behavior.
- [ ] Validate runtime behavior on at least one target GPU with validation layers enabled.

### Phase D: Adapter and Registry Integration
- [x] Add adapter and output mapping.
- [x] Register experiment in generated registry manifest.
- [x] Ensure CLI discoverability via `--experiment 04_sequential_indexing`.

### Phase E: Analysis and Artifact Pipeline
- [x] Add experiment-local `collect`, `analyze`, and `plot` scripts.
- [x] Add result-folder scaffold and output docs.
- [ ] Generate first complete dataset and finalize `results.md` interpretation.

### Phase F: Verification and Hardening
- [ ] Run warning-clean builds on current toolchain.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for touched translation units when compile database is available.
- [ ] Run unit tests and regression checks.

## 3. Acceptance Criteria Tracking
- [x] `--experiment 04_sequential_indexing` is discoverable and runnable from CLI.
- [x] Row-level output includes dispatch timing, end-to-end timing, throughput, GB/s, and correctness.
- [x] Result export includes median/p95 summaries per measured case.
- [ ] A reproducible dataset with charts/tables is generated under `experiments/04_sequential_indexing/results/`.
- [ ] `results.md` includes practical takeaways and explicit limitations.

## 4. Risks and Mitigations
- Risk: large sweep points may exceed dispatch or memory limits.
  Mitigation: runtime clamps by scratch size and `maxComputeWorkGroupCount[0]`.
- Risk: control overhead from high dispatch-count cases can dominate small problem sizes.
  Mitigation: keep dispatch GPU ms as the primary metric and compare by problem size.
- Risk: shader math drift between CPU and GPU reference checks.
  Mitigation: use deterministic, simple arithmetic with exact reference mapping.
