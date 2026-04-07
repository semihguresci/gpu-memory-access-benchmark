# Experiment 23 Results: Histogram Atomic Contention

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`30/30` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T14:05:31Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_140531Z_full_refresh_20260405.json)
- Sweep coverage: `6` variants x current configured problem sizes

## Key Measurements
- All `6` benchmark cases are represented in the refreshed export.
- At the largest tested `sample_count=16776960`, the fastest median GPU time came from `implementation=privatized_shared, distribution=hot_bin_90, local_size_x=256` at `20.637920 ms`. Median GB/s: `6.503`. Median throughput: `812919131.385`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [histogram atomic contention relative](./results/tables/histogram_atomic_contention_relative.csv)
- [histogram atomic contention stability](./results/tables/histogram_atomic_contention_stability.csv)
- [histogram atomic contention summary](./results/tables/histogram_atomic_contention_summary.csv)
- [histogram atomic contention estimated gbps](./results/charts/histogram_atomic_contention_estimated_gbps.png)
- [histogram atomic contention median gpu ms](./results/charts/histogram_atomic_contention_median_gpu_ms.png)
- [histogram atomic contention speedup vs global](./results/charts/histogram_atomic_contention_speedup_vs_global.png)
- [histogram atomic contention stability ratio](./results/charts/histogram_atomic_contention_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `2287.81%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
