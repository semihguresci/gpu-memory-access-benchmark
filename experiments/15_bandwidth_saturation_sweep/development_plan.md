# Experiment 15 Bandwidth Saturation Sweep: Codebase Development Plan

Date: 2026-03-27
Source spec: `docs/experiment_plans/15_bandwidth_saturation_sweep.md`

## 1. Scope and Current Status
This plan tracks implementation and workflow integration for Experiment 15 (`15_bandwidth_saturation_sweep`).

Current status:
- lecture-note experiment spec is present
- experiment-local README is present
- runtime architecture note is present
- runtime implementation, adapter wiring, shader, and analysis/plotting scripts are not present yet
- root workflow integration is not present yet in the manifest, build target list, or helper scripts
- no measured dataset or report is checked in under `experiments/15_bandwidth_saturation_sweep/results.md`

Primary implementation decisions for this experiment:
- implement Experiment 15 as a focused follow-up to Experiment 03, reusing the same simple contiguous memory modes
- keep the primary comparison on `read_only`, `write_only`, and `read_write_copy` while making problem size the main changing variable
- use a denser size sweep than Experiment 03 so plateau onset can be inferred from measured curves rather than guessed from sparse powers-of-two samples
- keep plateau estimation in the analysis layer so runtime behavior stays deterministic and easy to validate

## 2. Development Phases
### Phase A: Runtime Contract and Correctness
- [ ] Lock the primary mode set to `read_only`, `write_only`, and `read_write_copy`.
- [ ] Add the dense size sweep `1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024 MiB` with runtime clamping against `--size` and dispatch limits.
- [ ] Keep indexing fully contiguous, workgroup size fixed, and dispatch count fixed so size remains the main changing variable.
- [ ] Add deterministic validation for unchanged source contents, deterministic write-only outputs, and read-write copy outputs.
- [ ] Emit row-level output (`gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`, `correctness_pass`, `notes`) with size and bytes-per-element metadata stored in `notes`.
- [ ] Keep plateau heuristics in analysis scripts and record the realized sweep in generated tables or summaries.

### Phase B: Experiment Tooling
- [ ] Add experiment-local `collect_run.py`.
- [ ] Add experiment-local analysis and plotting scripts for saturation curves, plateau summary tables, and large-size variability.
- [ ] Add results/report scaffolding (`results.md`, `results/charts/`, `results/tables/`, `runs/`, and script docs).

### Phase C: Root Workflow Integration
- [ ] Add `15_bandwidth_saturation_sweep` to `cmake/experiments_manifest.cmake`.
- [ ] Add Experiment 15 sources to `CMakeLists.txt`.
- [ ] Add `15_bandwidth_saturation_sweep` to `scripts/run_experiment_data_collection.py`.
- [ ] Add `15_bandwidth_saturation_sweep` to `scripts/generate_experiment_artifacts.py`.
- [ ] Verify shader auto-compilation resolves the Experiment 15 shader outputs with unique basenames.

### Phase D: Validation and Hardening
- [ ] Configure with `cmake --preset windows-tests-vs`.
- [ ] Build release target with `cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments`.
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units when compile commands are available.
- [ ] Run tests and a benchmark smoke run for Experiment 15.

## 3. Acceptance Criteria Tracking
- [ ] `--experiment 15_bandwidth_saturation_sweep` is discoverable and runnable.
- [ ] Row-level output exists for the planned modes and dense size sweep and includes size metadata.
- [ ] Adapter reports failure on correctness mismatch or non-finite timing.
- [ ] Analysis emits a plateau summary table and at least one primary saturation chart.
- [ ] Root data collection and artifact-generation scripts support Experiment 15.
- [ ] Experiment-local `results.md` contains measured values, metadata, artifact links, and limitations.
