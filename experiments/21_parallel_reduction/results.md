# Experiment 21 Results: Parallel Reduction

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`60/60` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T14:04:22Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_140422Z_full_refresh_20260405.json)
- Sweep coverage: `2` variants x current configured problem sizes

## Key Measurements
- All `2` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=8388607`, the fastest median GPU time came from `variant=shared_tree` at `0.281248 ms`. Median GB/s: `119.305`. Median throughput: `29826370320.856`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [parallel reduction relative](./results/tables/parallel_reduction_relative.csv)
- [parallel reduction stability](./results/tables/parallel_reduction_stability.csv)
- [parallel reduction status overview](./results/tables/parallel_reduction_status_overview.csv)
- [parallel reduction summary](./results/tables/parallel_reduction_summary.csv)
- [parallel reduction gbps](./results/charts/parallel_reduction_gbps.png)
- [parallel reduction median gpu ms](./results/charts/parallel_reduction_median_gpu_ms.png)
- [parallel reduction speedup vs global atomic](./results/charts/parallel_reduction_speedup_vs_global_atomic.png)
- [parallel reduction stability ratio](./results/charts/parallel_reduction_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `2.87%`, so the sweep does not show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
