# Experiment 16 Results: Shared Memory Tiling

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`40/40` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T14:00:19Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_140019Z_full_refresh_20260405.json)
- Sweep coverage: `8` variants x current configured problem sizes

## Key Measurements
- All `8` benchmark cases are represented in the refreshed export.
- At the largest tested `logical_elements=4194048`, the fastest median GPU time came from `implementation=shared_tiled, reuse_radius=16, local_size_x=256` at `1.503008 ms`. Median GB/s: `23.719`. Median throughput: `2790436245.183`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [shared memory tiling relative](./results/tables/shared_memory_tiling_relative.csv)
- [shared memory tiling stability](./results/tables/shared_memory_tiling_stability.csv)
- [shared memory tiling summary](./results/tables/shared_memory_tiling_summary.csv)
- [shared memory tiling estimated gbps](./results/charts/shared_memory_tiling_estimated_gbps.png)
- [shared memory tiling median gpu ms](./results/charts/shared_memory_tiling_median_gpu_ms.png)
- [shared memory tiling speedup vs direct](./results/charts/shared_memory_tiling_speedup_vs_direct.png)
- [shared memory tiling stability ratio](./results/charts/shared_memory_tiling_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `3.47%`, so the sweep does not show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
