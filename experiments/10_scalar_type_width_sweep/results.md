# Experiment 10 Results: Scalar Type Width Sweep

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`175/175` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:55:55Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135555Z_full_refresh_20260405.json)
- Sweep coverage: `5` variants x current configured problem sizes

## Key Measurements
- All `5` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=8388608`, the fastest median GPU time came from `variant=u8` at `0.754944 ms`. Median GB/s: `22.223`. Median throughput: `11111563241.777`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [scalar type width sweep largest size comparison](./results/tables/scalar_type_width_sweep_largest_size_comparison.csv)
- [scalar type width sweep scaling by size](./results/tables/scalar_type_width_sweep_scaling_by_size.csv)
- [scalar type width sweep speedup crossover](./results/tables/scalar_type_width_sweep_speedup_crossover.csv)
- [scalar type width sweep speedup vs fp32 by size](./results/tables/scalar_type_width_sweep_speedup_vs_fp32_by_size.csv)
- [scalar type width sweep stability overview](./results/tables/scalar_type_width_sweep_stability_overview.csv)
- [scalar type width sweep status overview](./results/tables/scalar_type_width_sweep_status_overview.csv)
- [scalar type width sweep error metrics](./results/charts/scalar_type_width_sweep_error_metrics.png)
- [scalar type width sweep gpu ms per million elements](./results/charts/scalar_type_width_sweep_gpu_ms_per_million_elements.png)
- [scalar type width sweep median gbps](./results/charts/scalar_type_width_sweep_median_gbps.png)
- [scalar type width sweep median gpu ms](./results/charts/scalar_type_width_sweep_median_gpu_ms.png)
- [scalar type width sweep p95 to median ratio](./results/charts/scalar_type_width_sweep_p95_to_median_ratio.png)
- [scalar type width sweep speedup vs fp32](./results/charts/scalar_type_width_sweep_speedup_vs_fp32.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `391.97%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
