# Experiment 15 Bandwidth Saturation Sweep: Implementation Plan

Date: 2026-03-27

## Goal
Implement a correctness-first Vulkan compute experiment that sweeps simple contiguous memory modes across a dense range
of problem sizes so the onset of sustained practical bandwidth can be measured rather than inferred from sparse points.

Primary configuration:
- Experiment ID: `15_bandwidth_saturation_sweep`
- Mode sweep: `read_only`, `write_only`, `read_write_copy`
- Size sweep: `1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024 MiB`, runtime-clamped by
  per-buffer budget and `maxComputeWorkGroupCount[0]`
- Workgroup size: `256`
- Dispatch count: `1` per timed sample
- Timing outputs: dispatch GPU ms primary, end-to-end ms supporting
- Buffer budget: allocate source, destination, and staging buffers for the largest valid point under the existing
  `--size` cap
- Accuracy outputs: deterministic source and sentinel patterns with exact validation per mode
- Measurement note: `throughput` is the primary rate metric in logical elements per second; `gbps` is retained as the
  logical bytes-per-second metric based on mode-specific bytes moved; plateau onset is derived from summarized curves in
  analysis

## Variant Contract
- `read_only`: contiguous `readonly volatile` load with no output write
- `write_only`: contiguous deterministic write `dst[id] = float(id)`
- `read_write_copy`: contiguous copy `dst[id] = src[id]`

All primary variants use identical contiguous 1D indexing and the same workgroup size. The first implementation should
keep size as the main changing variable.

## Scope
- Run target: separate overhead-bound and sustained-throughput regions for the three simple memory modes
- Logical element count: derived from each size point as `size_bytes / sizeof(float)`
- Reporting: per-mode median and p95 timing, logical throughput, logical GB/s, and plateau-onset summary tables
- Validation policy: fail the run on any mismatch before emitting benchmark success

## File Changes
New files:
- `include/experiments/bandwidth_saturation_sweep_experiment.hpp`
- `src/experiments/bandwidth_saturation_sweep_experiment.cpp`
- `src/experiments/adapters/bandwidth_saturation_sweep_adapter.cpp`
- `shaders/15_bandwidth_saturation_sweep/15_bandwidth_read_only.comp`
- `shaders/15_bandwidth_saturation_sweep/15_bandwidth_write_only.comp`
- `shaders/15_bandwidth_saturation_sweep/15_bandwidth_read_write_copy.comp`
- `experiments/15_bandwidth_saturation_sweep/architecture.md`
- `experiments/15_bandwidth_saturation_sweep/development_plan.md`
- `experiments/15_bandwidth_saturation_sweep/implementation_plan.md`
- `experiments/15_bandwidth_saturation_sweep/results.md`
- `experiments/15_bandwidth_saturation_sweep/scripts/collect_run.py`
- `experiments/15_bandwidth_saturation_sweep/scripts/analyze_bandwidth_saturation_sweep.py`
- `experiments/15_bandwidth_saturation_sweep/scripts/plot_results.py`

Touched files:
- `cmake/experiments_manifest.cmake`
- `CMakeLists.txt`
- `scripts/run_experiment_data_collection.py`
- `scripts/generate_experiment_artifacts.py`
- `docs/experiment_plans/15_bandwidth_saturation_sweep.md`
- `experiments/15_bandwidth_saturation_sweep/experiment_plan.md`

## Work Packages
### 1. Runtime Contract and Sweep Generation
- [ ] Add Experiment 15 config and output contract.
- [ ] Add a dense size sweep generator with explicit clamp rules for buffer capacity and dispatch limits.
- [ ] Keep bytes-per-element fixed per mode and record size metadata in row notes.
- [ ] Record row-note metadata such as `size_bytes`, `size_mib`, `bytes_per_element`, `logical_elements`,
  `local_size_x`, and `group_count_x`.

### 2. Vulkan Runtime and Shaders
- [ ] Add buffer, descriptor, and pipeline setup/teardown for the three contiguous memory modes.
- [ ] Add a source buffer, destination buffer, and staging buffer that are seeded before warmup and timed iterations.
- [ ] Resolve and load the `15_bandwidth_*.comp.spv` shaders through the existing shader-path helper.
- [ ] Keep each kernel path explicit and bounds-safe with no extra synchronization or control flow.
- [ ] Preserve reverse-order teardown and handle reset rules for all Vulkan objects.

### 3. Correctness and Metrics
- [ ] Reuse the deterministic source-pattern and sentinel validation strategy from Experiment 03.
- [ ] Validate source invariance for `read_only`, exact write pattern for `write_only`, and exact copy results for
  `read_write_copy`.
- [ ] Emit row-level metrics for each timed iteration and mode-size combination.
- [ ] Capture logical bytes processed so the derived `gbps` column stays comparable across modes.
- [ ] Summarize timing samples with the existing benchmark runner utilities.

### 4. Workflow Integration
- [ ] Add Experiment 15 to the manifest and build target list.
- [ ] Add Experiment 15 to the root data-collection helper.
- [ ] Add Experiment 15 to the root artifact-generation helper.
- [ ] Add experiment-local run collection, analysis, and plotting scripts.
- [ ] Update experiment-local reporting scaffolding so charts and tables have deterministic names and plateau summary
  outputs.

### 5. Verification Gates
- [ ] Build release target (`tests-vs-release` preset).
- [ ] Run `clang-format` on touched C++ files.
- [ ] Run `clang-tidy` for modified translation units.
- [ ] Run tests and a benchmark smoke pass for Experiment 15.
