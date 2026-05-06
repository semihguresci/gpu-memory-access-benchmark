# Experiment 42 Results: Async Compute Overlap

## Run Status
- Benchmark status: latest `20260428_203111Z_2` collection completed; all `12/12` benchmark cases in the refreshed sweep report `correctness_pass_rate=1.0`
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `2` variants x `6` problem sizes, `5` timed iterations, `2` warmup iterations
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T20:31:11Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_203111Z_2.json)

## Key Measurements
- `fused_overlap_proxy` is faster than `serial_no_overlap` at `5/6` tested sizes, but the spread stays small: from `0.31%` slower at `4194304` elements (`3.047424 ms` vs `3.038112 ms`) to `2.44%` faster at `33554432` elements (`26.447232 ms` vs `27.107712 ms`)
- The largest absolute median GPU-time reduction appears at `67108864` elements, where `fused_overlap_proxy` reduces median time by `1.077312 ms` (`51.182720 ms` vs `52.260032 ms`) and raises median estimated bandwidth from `20.546138` to `20.978600 GB/s`
- Stability is mostly tight once the workload is large; the highest `p95/median` ratios are `1.732405` for `serial_no_overlap` at `1048576` elements and `1.518429` for `fused_overlap_proxy` at `4194304` elements

## Artifact Links
- [Latest runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_203111Z_2.json)
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [async compute overlap relative](./results/tables/async_compute_overlap_relative.csv)
- [async compute overlap stability](./results/tables/async_compute_overlap_stability.csv)
- [async compute overlap summary](./results/tables/async_compute_overlap_summary.csv)
- [async compute overlap median gpu ms](./results/charts/async_compute_overlap_ms.png)
- [async compute overlap relative](./results/charts/async_compute_overlap_relative.png)
- [async compute overlap stability](./results/charts/async_compute_overlap_stability.png)
- [async compute overlap throughput](./results/charts/async_compute_overlap_throughput.png)

## Interpretation
- On this GPU, the single-queue `fused_overlap_proxy` does not produce a step-change versus the serial baseline; measured differences stay between `0.31%` slower and `2.44%` faster across the current size sweep
- The refreshed data is consistent with modest savings from reduced dispatch or barrier overhead at larger sizes, not with a large overlap effect

## Limitations
- Results come from one GPU and driver stack
- This experiment compares `serial_no_overlap` against a fused proxy kernel; it is not a direct measurement of independent queue overlap
- Several observed deltas are close to the measured run-to-run spread at smaller and mid-sized points, so sub-`3%` differences should be treated as directional
