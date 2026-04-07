# Experiment 25 Results: Spatial Binning Clustered Culling Capstone

## Run Status
- Benchmark status: implementation added, but no fresh Experiment 25 collection was run in this change.
- Scope: 1D spatial binning/list construction capstone, not full clustered culling.
- Validation mode: per-bin count match plus sorted per-bin entity-id comparison.
- Strategies: `global_append` and `coherent_append`.
- Distributions: `uniform_sparse`, `uniform_dense`, and `clustered`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [summary table](./results/tables/spatial_binning_clustered_culling_capstone_summary.csv)
- [relative table](./results/tables/spatial_binning_clustered_culling_capstone_relative.csv)
- [stability table](./results/tables/spatial_binning_clustered_culling_capstone_stability.csv)
- [median GPU ms chart](./results/charts/spatial_binning_clustered_culling_capstone_median_gpu_ms.png)
- [estimated GB/s chart](./results/charts/spatial_binning_clustered_culling_capstone_estimated_gbps.png)
- [speedup chart](./results/charts/spatial_binning_clustered_culling_capstone_speedup_vs_global.png)
- [stability chart](./results/charts/spatial_binning_clustered_culling_capstone_stability_ratio.png)

## Interpretation
- The capstone is intentionally scoped to deterministic spatial binning so correctness is reviewable before expanding toward full clustered culling.
- `coherent_append` is implemented as host-sorted, bin-coherent input feeding the same append kernel; any measured delta therefore reflects input coherence rather than a different shader pipeline.

## Limitations
- No fresh measurement data is linked yet; regenerate artifacts after Experiment 25 is registered and collected.
- The benchmark models one-dimensional binning and append-list construction only.
- Output ordering is intentionally treated as nondeterministic; comparisons are based on counts and per-bin contents.
