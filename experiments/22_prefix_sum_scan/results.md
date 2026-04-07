# Experiment 22 Results: Prefix Sum Scan

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`20/20` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T14:04:22Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_140422Z_full_refresh_20260405.json)
- Sweep coverage: `4` variants x current configured problem sizes

## Key Measurements
- All `4` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=65536`, the fastest median GPU time came from `variant=items_per_thread_1` at `0.062624 ms`. Median GB/s: `16.809`. Median throughput: `3139499233.521`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [prefix sum scan relative](./results/tables/prefix_sum_scan_relative.csv)
- [prefix sum scan stability](./results/tables/prefix_sum_scan_stability.csv)
- [prefix sum scan summary](./results/tables/prefix_sum_scan_summary.csv)
- [prefix sum scan effective gbps](./results/charts/prefix_sum_scan_effective_gbps.png)
- [prefix sum scan median gpu ms](./results/charts/prefix_sum_scan_median_gpu_ms.png)
- [prefix sum scan speedup vs baseline](./results/charts/prefix_sum_scan_speedup_vs_baseline.png)
- [prefix sum scan stability ratio](./results/charts/prefix_sum_scan_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `7.61%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
