# Experiment 14 Results: Read Reuse Cache Locality

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`25/25` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:57:39Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135739Z_full_refresh_20260405.json)
- Sweep coverage: `5` variants x current configured problem sizes

## Key Measurements
- All `5` benchmark cases are represented in the refreshed export.
- At the largest tested `logical_elements=3355442`, the fastest median GPU time came from `variant=reuse_distance_1` at `1.570304 ms`. Median GB/s: `25.642`. Median throughput: `2136810451.989`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [read reuse cache locality locality](./results/tables/read_reuse_cache_locality_locality.csv)
- [read reuse cache locality relative](./results/tables/read_reuse_cache_locality_relative.csv)
- [read reuse cache locality stability](./results/tables/read_reuse_cache_locality_stability.csv)
- [read reuse cache locality summary](./results/tables/read_reuse_cache_locality_summary.csv)
- [read reuse cache locality median gbps](./results/charts/read_reuse_cache_locality_median_gbps.png)
- [read reuse cache locality median gpu ms](./results/charts/read_reuse_cache_locality_median_gpu_ms.png)
- [read reuse cache locality speedup vs full span](./results/charts/read_reuse_cache_locality_speedup_vs_full_span.png)
- [read reuse cache locality stability ratio](./results/charts/read_reuse_cache_locality_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `32.65%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
