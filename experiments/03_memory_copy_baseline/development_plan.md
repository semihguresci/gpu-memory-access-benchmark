# Experiment 03 Memory Copy Baseline: Codebase Development Plan

Date: 2026-03-15  
Source spec: `docs/experiment_plans/03_memory_copy_baseline.md`

## 1. Scope and Current Status
This plan tracks implementation of canonical Experiment 03 in the current Vulkan benchmark harness.

Current status:
- runtime implementation for `03_memory_copy_baseline` is integrated
- experiment is registered in `cmake/experiments_manifest.cmake`
- shaders and adapter wiring are present
- analysis script scaffolding is present, but measured data is pending

## 2. Development Phases
### Phase A: Design Freeze and Constraints
- [x] Lock mode set: `read_only`, `write_only`, `read_write_copy`.
- [x] Lock fixed local size strategy and clamp behavior.
- [x] Lock sweep range policy: powers-of-two from 1 MiB to 1 GiB (runtime-clamped).
- [x] Lock correctness strategy per mode.

### Phase B: Shader and Pipeline Preparation
- [x] Add dedicated shaders for each memory mode.
- [x] Validate naming convention and runtime lookup path.
- [ ] Validate shader binaries across all configured build presets.

### Phase C: Core Runtime Implementation
- [x] Add experiment API and implementation in `include/experiments/` and `src/experiments/`.
- [x] Implement explicit Vulkan ownership and reverse-order teardown.
- [x] Add warmup/timed execution loops with row-level output.
- [x] Add deterministic correctness checks for all modes.
- [ ] Validate runtime behavior on at least one target GPU with validation layers enabled.

### Phase D: Adapter and Registry Integration
- [x] Add adapter and output mapping.
- [x] Register experiment in generated registry manifest.
- [x] Ensure CLI discoverability via `--experiment 03_memory_copy_baseline`.

### Phase E: Analysis and Artifact Pipeline
- [x] Add experiment-local `collect`, `analyze`, and `plot` scripts.
- [x] Add result-folder scaffold and output docs.
- [ ] Generate first complete dataset and finalize `results.md` interpretation.

### Phase F: Verification and Hardening
- [ ] Run warning-clean builds on current toolchain.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for touched translation units when compile commands are available.
- [ ] Run unit tests and regression checks.

## 3. Acceptance Criteria Tracking
- [x] `--experiment 03_memory_copy_baseline` is discoverable and runnable from CLI.
- [x] All three modes emit row-level timing and correctness fields.
- [x] Result export includes median/p95 summaries per measured case.
- [ ] A reproducible dataset with charts/tables is generated under `experiments/03_memory_copy_baseline/results/`.
- [ ] `results.md` includes practical takeaways and explicit limitations.

## 4. Risks and Mitigations
- Risk: large sweep points may exceed memory budget on some GPUs.
  Mitigation: runtime clamps by configured scratch size and dispatch limits.
- Risk: read-only mode may be optimized too aggressively.
  Mitigation: use `volatile` storage reads and validate source-buffer stability.
- Risk: interpretation confusion from mixed bytes-per-mode.
  Mitigation: encode mode-specific effective GB/s in row output and analysis tables.
