# Experiment 13 Results: Scatter Access Pattern

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`15/15` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:57:13Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135713Z_full_refresh_20260405.json)
- Sweep coverage: `3` variants x current configured problem sizes

## Key Measurements
- All `3` benchmark cases are represented in the refreshed export.
- At the largest tested `logical_elements=4194304`, the fastest median GPU time came from `variant=random_collision_x4, distribution=random_collision_x4` at `24.503360 ms`. Median GB/s: `2.054`. Median throughput: `171172606.532`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [scatter access pattern contention](./results/tables/scatter_access_pattern_contention.csv)
- [scatter access pattern relative](./results/tables/scatter_access_pattern_relative.csv)
- [scatter access pattern stability](./results/tables/scatter_access_pattern_stability.csv)
- [scatter access pattern summary](./results/tables/scatter_access_pattern_summary.csv)
- [scatter access pattern median gbps](./results/charts/scatter_access_pattern_median_gbps.png)
- [scatter access pattern median gpu ms](./results/charts/scatter_access_pattern_median_gpu_ms.png)
- [scatter access pattern slowdown vs unique permutation](./results/charts/scatter_access_pattern_slowdown_vs_unique_permutation.png)
- [scatter access pattern stability ratio](./results/charts/scatter_access_pattern_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `124.46%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
