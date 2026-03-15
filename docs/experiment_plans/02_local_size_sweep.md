# Experiment 02: Local Size Sweep

Date: 2026-03-14  
Canonical experiment ID: `02_local_size_sweep`  
Baseline dependency: `01_dispatch_basics`

## 1. Lecture Focus
- Concept: Workgroup size (`local_size_x`) as a first-order control for occupancy, scheduling granularity, and memory transaction shape.
- Why this matters: Poor local-size choices can leave GPU cores underutilized or trigger unnecessary register/shared-resource pressure.
- Central question: Which legal `local_size_x` values provide the best and most stable throughput for a fixed kernel and fixed GPU?

## 2. Learning Objectives
By the end of this experiment, you should be able to:
- explain why local size interacts with warp or wave execution width
- design a Vulkan sweep that isolates local-size effects from unrelated variables
- interpret timing stability (median vs p95 spread) alongside peak throughput
- derive a practical local-size recommendation for follow-on experiments

## 3. Scope and Non-Goals
In scope:
- a controlled local-size sweep for one contiguous read-modify-write compute kernel
- correctness validation for every measured point
- machine-readable outputs for charting and cross-run comparison

Out of scope:
- algorithmic changes between points
- vendor-specific occupancy APIs
- automatic tuning integrated into runtime selection logic

## 4. Theory Primer and Predictions
- `local_size_x` controls threads per workgroup, which determines how many workgroups can be resident concurrently.
- Very small groups often increase scheduling overhead and reduce occupancy.
- Very large groups can reduce active workgroups per SM/CU due to per-group resource pressure.
- Best performance is expected near subgroup-aligned values and can vary by architecture.

Falsifiable predictions:
- H1: At least one middle-range local size (`64`, `128`, or `256`) beats both extremes (`32` and `1024`) by >= 5% median throughput.
- H2: The winning local size remains in the top 2 across most problem sizes.
- H3: p95 instability is higher at extreme local sizes than at the best-performing local size.

## 5. Experimental Design
### Independent variables
- `local_size_x`: candidate set `32, 64, 128, 256, 512, 1024` filtered by device legality.
- `problem_size`: powers of two from `2^14` to `2^24`, clamped by `--size` and dispatch limits.
- `variant`: `contiguous_write` (primary), `noop` (control path for dispatch overhead context).

### Controlled variables
- identical kernel logic per variant (only `local_size_x` changes)
- identical descriptor layout, pipeline layout, and buffer usage flags
- identical warmup/timed iteration counts from `BenchmarkRunner`
- identical queue family, timestamp mechanism, and synchronization phases
- fixed dispatch repetition count per measured point (default `dispatch_count = 256`)

### Workload design
- Primary kernel: contiguous read-modify-write over `float` storage buffer.
- Control kernel: no-op dispatch path to quantify launch/scheduling baseline.
- Data init uses deterministic sentinel values to keep validation simple and robust.

## 6. Device Capability and Legality Rules
Each candidate `local_size_x` is legal only if all checks pass:
- `local_size_x <= maxComputeWorkGroupSize[0]`
- `local_size_x <= maxComputeWorkGroupInvocations`
- computed `group_count_x <= maxComputeWorkGroupCount[0]`

Runtime behavior for illegal candidates:
- skip candidate with explicit reason in logs and metadata
- never attempt pipeline creation/dispatch for illegal sizes
- fail run only if no candidate survives filtering

## 7. Measurement Protocol
- Warmup: `--warmup` iterations per point (default 5).
- Timed: `--iterations` iterations per point (default 20).
- GPU timer: timestamp queries around dispatch stage; also capture upload/readback stage timings.
- Host timer: end-to-end wall time around upload + dispatch + readback.
- Primary report statistic: median `gpu_ms`.
- Stability indicator: p95 `gpu_ms`.

Derived metrics:
- `throughput = (problem_size * dispatch_count) / (gpu_ms / 1000.0)`
- `gbps = (problem_size * dispatch_count * sizeof(float)) / (gpu_ms * 1e6)`
- optional relative speedup vs baseline local size (`64`): `median_ms(64) / median_ms(ls)`

## 8. Data to Capture
Required row-level fields (existing schema):
- `experiment_id`, `variant`, `problem_size`, `dispatch_count`, `iteration`
- `gpu_ms`, `end_to_end_ms`, `throughput`, `gbps`
- `correctness_pass`, `notes`

Recommended encoding conventions:
- `variant` format: `contiguous_write_ls64`, `noop_ls128`, etc.
- `notes` tags: `local_size_x=<N>`, `subgroup_size=<S>`, `legality=pass`

Summary-level fields:
- average/min/max/median/p95/sample_count per `(variant, problem_size, dispatch_count)`

## 9. Analysis Plan
1. Build pivot table of median `gpu_ms` by `problem_size x local_size_x`.
2. Plot local-size sweep curves for representative problem sizes.
3. Plot throughput bars at largest stable problem size.
4. Compute speedup ratios relative to `local_size_x=64`.
5. Rank local sizes by geometric-mean speedup across all valid problem sizes.
6. Flag unstable points where `(p95 - median) / median > 0.15`.

Interpretation rules:
- do not compare points with failed correctness
- separate dispatch-overhead effects (noop) from memory-work effects (contiguous write)
- report both absolute performance and stability
- document legal-range constraints before recommending a winner

## 10. Common Failure Modes
- candidate local size exceeds device limits
- subgroup assumptions incorrectly treated as fixed across vendors
- non-finite stage timings from synchronization/timestamp misuse
- correctness mismatches caused by bad barriers or stale buffer contents
- excessive case count causing impractically long run time

## 11. Deliverables
Minimum artifact set:
- `benchmark_results.json` including per-iteration rows
- one CSV pivot table (`problem_size x local_size_x`)
- one local-size vs median-ms chart
- one local-size vs throughput chart
- short analysis notes with recommendation and explicit limitations

Recommended additional artifacts:
- legality report per device
- stability chart (median and p95 spread)
- cross-device summary if multiple run snapshots exist

## 12. Exit Criteria
Experiment 02 is considered complete when:
- at least 4 legal local-size candidates were measured successfully
- all measured rows pass correctness checks
- median/p95 summaries exist for every measured point
- a concrete local-size recommendation is documented with caveats
- outputs are reproducible by rerunning with the same CLI parameters

## 13. Follow-Up Link
Use the selected local-size recommendation as the default for:
- `03_memory_copy_baseline` where applicable
- `04_sequential_indexing`
- later access-pattern experiments unless explicitly overridden
