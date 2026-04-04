# Experiment 01 Results: Dispatch Basics

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`880/880` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 4M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:08:09Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_140809Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_140809Z_full_refresh_20260329.json)
- Sweep coverage: `2` variants x `11` problem sizes x `8` dispatch counts

## Key Measurements
- At the largest tested problem size (`problem_size=1048576`), the lowest per-configuration GPU time still came from single-dispatch runs: `contiguous_write`, `dispatch_count=1` at `0.019968 ms` and `noop`, `dispatch_count=1` at `0.021248 ms`.
- Peak aggregate throughput at that same size again came from batched dispatches: `contiguous_write`, `dispatch_count=1024` reached `80.381 GElem/s` (`321.526 GB/s`) and `noop`, `dispatch_count=1024` reached `77.641 GElem/s` (`310.563 GB/s`).
- The write-over-noop GPU-time ratio tightened from `1.113x` at `dispatch_count=1` to `1.047x` at `dispatch_count=64`, so fixed dispatch overhead still dominated more of the total cost as batching increased.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Current summary table](./results/tables/dispatch_basics_summary.csv)
- [Best dispatch table](./results/tables/dispatch_basics_best_dispatch.csv)
- [Write/noop ratio table](./results/tables/dispatch_basics_ratio_summary.csv)
- [Representative points chart](./results/charts/dispatch_basics_representative_points.png)
- [Time and throughput chart](./results/charts/dispatch_basics_time_throughput.png)

## Interpretation
- This full-refresh snapshot keeps the same core split between latency tuning and throughput tuning: `dispatch_count=1` minimizes raw GPU time for a single configuration, while larger dispatch batches maximize aggregate work rate.
- At large sizes, `contiguous_write` and `noop` stay close. On this GPU, once enough work is batched, the extra sequential store is still a second-order cost relative to dispatch mechanics.

## Limitations
- Results come from one GPU and driver stack.
- This benchmark isolates dispatch structure, not application-level queue contention or multi-pass pipelines.
