# Experiment 38 Results: Frustum vs Clustered Culling

## Run Status
- Benchmark status: latest collection completed (`120/120` row correctness pass)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `5` timed iterations, `2` warmup iterations; current sweep keeps `cluster_count=16` and `local_size_x=256`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T17:18:23Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_171823Z.json)
- Sweep coverage: `4` variants x `6` problem sizes across `2` scene distributions

## Key Measurements
- All `24` benchmark cases are represented in the current export.
- At the largest tested `entity_count=8388608`, the fastest median GPU time came from `variant=frustum_direct_wide_scene` at `16.818304 ms`. Median GB/s: `10.279`. Median throughput: `498778473.739`.
- At the same size, `clustered_culling_wide_scene` measured `128.162336 ms` and `clustered_culling_center_clustered` measured `153.588544 ms`, which is `7.620408x` and `7.266834x` slower than the matched frustum baseline for each distribution.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [frustum vs clustered culling relative](./results/tables/frustum_vs_clustered_culling_relative.csv)
- [frustum vs clustered culling stability](./results/tables/frustum_vs_clustered_culling_stability.csv)
- [frustum vs clustered culling summary](./results/tables/frustum_vs_clustered_culling_summary.csv)
- [frustum vs clustered culling median gpu ms](./results/charts/frustum_vs_clustered_culling_ms.png)
- [frustum vs clustered culling relative ratios](./results/charts/frustum_vs_clustered_culling_relative.png)
- [frustum vs clustered culling stability ratio](./results/charts/frustum_vs_clustered_culling_stability.png)
- [frustum vs clustered culling throughput](./results/charts/frustum_vs_clustered_culling_throughput.png)

## Interpretation
- In the current sweep, direct frustum append stayed faster than clustered bin counting in both scene distributions.
- The gap is not limited to the small cases: at `entity_count=8388608`, clustered culling remained about `7.3x` to `7.6x` slower than the matching frustum baseline.

## Limitations
- Results come from one GPU and driver stack.
- The current sweep fixes `cluster_count=16` and `local_size_x=256`; different cluster granularity or workgroup sizing may change the tradeoff.
- Reported GB/s and throughput values are derived from experiment-specific estimated traffic definitions; use median GPU time as the primary ranking signal.
