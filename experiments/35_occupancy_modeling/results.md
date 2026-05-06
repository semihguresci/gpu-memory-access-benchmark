# Experiment 35 Results: Occupancy Modeling

## Run Status
- Benchmark status: latest `20260428_171131Z` collection completed (`60/60` row correctness pass across `12` benchmark cases)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T17:11:31Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_171131Z.json)
- Sweep coverage: `3` variants x `4` problem sizes (`655360` through `41943040` elements)

## Key Measurements
- All `12` benchmark cases are represented in the refreshed export.
- `high_smem` recorded the lowest median GPU time at every tested size. At `element_count=10485760`, it measured `3.709216 ms` versus `3.911936 ms` for `low_smem` and `3.907616 ms` for `medium_smem`. Median GB/s: `22.615582`, `21.443623`, and `21.467329`.
- The largest stability outlier in the current table is `high_smem` at `element_count=2621440`, where p95-to-median GPU time reached `1.603214` despite a median advantage of only about `0.53%` over `low_smem`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [occupancy modeling relative](./results/tables/occupancy_modeling_relative.csv)
- [occupancy modeling stability](./results/tables/occupancy_modeling_stability.csv)
- [occupancy modeling summary](./results/tables/occupancy_modeling_summary.csv)
- [occupancy modeling gbps](./results/charts/occupancy_modeling_gbps.png)
- [occupancy modeling gpu ms](./results/charts/occupancy_modeling_gpu_ms.png)
- [occupancy modeling slowdown](./results/charts/occupancy_modeling_slowdown.png)
- [occupancy modeling stability](./results/charts/occupancy_modeling_stability.png)

## Interpretation
- On this GPU and with the current kernels, the higher shared-memory configuration produced the best medians across the tested sizes, but the win is modest rather than dramatic.
- The stability table matters here: some of the median differences are only a few percent, and `high_smem` shows a noticeably worse tail at `2621440` elements.

## Limitations
- Results come from one GPU and driver stack.
- This report uses the experiment's current `high_smem`, `medium_smem`, and `low_smem` variants only; it does not isolate every occupancy-related factor independently.
- Median and GB/s rankings should be interpreted alongside the stability table, especially where the median gaps are small.
