# Experiment 02 Results: Local Size Sweep

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`480/480` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:48:05Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_134805Z_full_refresh_20260405.json)
- Sweep coverage: `12` variants x current configured problem sizes

## Key Measurements
- All `12` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=2097152`, the fastest median GPU time came from `local_size_x=512` at `0.013728 ms`. Median GB/s: `0.000`. Median throughput: `0.000`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [local size sweep best local size](./results/tables/local_size_sweep_best_local_size.csv)
- [local size sweep gpu ms pivot](./results/tables/local_size_sweep_gpu_ms_pivot.csv)
- [local size sweep local size ranking](./results/tables/local_size_sweep_local_size_ranking.csv)
- [local size sweep multi run summary](./results/tables/local_size_sweep_multi_run_summary.csv)
- [local size sweep operation ratio](./results/tables/local_size_sweep_operation_ratio.csv)
- [local size sweep operation ratio summary](./results/tables/local_size_sweep_operation_ratio_summary.csv)
- [benchmark summary](./results/charts/benchmark_summary.png)
- [local size sweep gpu ms vs local size](./results/charts/local_size_sweep_gpu_ms_vs_local_size.png)
- [local size sweep operation ratio summary](./results/charts/local_size_sweep_operation_ratio_summary.png)
- [local size sweep speedup vs ls64](./results/charts/local_size_sweep_speedup_vs_ls64.png)
- [local size sweep status overview](./results/charts/local_size_sweep_status_overview.png)
- [local size sweep test setup](./results/charts/local_size_sweep_test_setup.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `414.69%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
