# Experiment 03 Results: Memory Copy Baseline

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`105/105` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 64M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:10:22Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_141022Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_141022Z_full_refresh_20260329.json)
- Sweep coverage: `3` variants x `7` problem sizes

## Key Measurements
- At the largest problem size in the current full-refresh run (`problem_size=16777216`), `read_only` reached `0.078144 ms` and `858.785 GB/s`.
- `write_only` measured `0.177024 ms` and `379.095 GB/s`, which is `2.265x` slower than `read_only` in GPU time.
- `read_write_copy` measured `0.320704 ms` and `418.510 GB/s`, which is `4.104x` slower than `read_only` in GPU time.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Current summary table](./results/tables/memory_copy_baseline_summary.csv)
- [Status overview table](./results/tables/memory_copy_baseline_status_overview.csv)
- [GPU time chart](./results/charts/memory_copy_baseline_gpu_ms_vs_size.png)
- [GB/s chart](./results/charts/memory_copy_baseline_gbps_vs_size.png)

## Interpretation
- `read_only` remains the upper-bound path for this benchmark, while `write_only` shows the expected store-side penalty and `read_write_copy` absorbs the longest elapsed time because it combines both directions of traffic.
- The `GB/s` values in this experiment represent different byte flows per variant, so elapsed GPU time is still the cleaner apples-to-apples comparison across the three operation types.

## Limitations
- Results come from one GPU and driver stack.
- The current headline dataset in `benchmark_results.json` tops out at `problem_size=16777216`; older archived runs cover larger sizes but are not the report baseline here.
