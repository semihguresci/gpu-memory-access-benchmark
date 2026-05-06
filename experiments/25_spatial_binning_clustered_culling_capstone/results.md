# Experiment 25 Results: Spatial Binning Clustered Culling Capstone

## Run Status
- Benchmark status: latest `20260428_195247Z_2` collection completed (`30/30` row correctness pass across `6` cases)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T19:52:47Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_195247Z_2.json)
- Sweep coverage: `2` strategies x `3` distributions at `8134400` entities and `64` bins

## Key Measurements
- All `6` benchmark cases report `correctness_pass_rate=1.000000`.
- The fastest median GPU time was `strategy=coherent_append, distribution=uniform_sparse` at `14.444608 ms`. Median GB/s: `9.010310`. Median throughput: `563144392.703492`.
- `coherent_append` beat `global_append` for every tested distribution, with speedups from `9.462647x` (`clustered`) to `13.454893x` (`uniform_sparse`).

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [spatial binning clustered culling capstone summary](./results/tables/spatial_binning_clustered_culling_capstone_summary.csv)
- [spatial binning clustered culling capstone relative](./results/tables/spatial_binning_clustered_culling_capstone_relative.csv)
- [spatial binning clustered culling capstone stability](./results/tables/spatial_binning_clustered_culling_capstone_stability.csv)
- [spatial binning clustered culling capstone estimated gbps](./results/charts/spatial_binning_clustered_culling_capstone_estimated_gbps.png)
- [spatial binning clustered culling capstone median gpu ms](./results/charts/spatial_binning_clustered_culling_capstone_median_gpu_ms.png)
- [spatial binning clustered culling capstone speedup vs global](./results/charts/spatial_binning_clustered_culling_capstone_speedup_vs_global.png)
- [spatial binning clustered culling capstone stability ratio](./results/charts/spatial_binning_clustered_culling_capstone_stability_ratio.png)

## Interpretation
- The coherent-input path stayed in a narrow `14.444608` to `15.072992 ms` band, while `global_append` ranged from `142.630400` to `194.350656 ms` across the same distributions.
- Both variants use the same append kernel, so the refreshed gap is specific to the current host-side input ordering setup rather than to a different GPU pipeline.

## Limitations
- Results come from one GPU/driver stack, one entity count (`8134400`), and `64` bins.
- This capstone measures 1D spatial binning and list construction, not full clustered culling.
- Correctness is based on per-bin counts plus sorted per-bin entity-id comparison; insertion order is not part of the oracle.
