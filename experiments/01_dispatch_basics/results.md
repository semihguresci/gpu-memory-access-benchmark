# Experiment 01 Results: Dispatch Basics

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`720/720` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:47:43Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_134743Z_full_refresh_20260405.json)
- Sweep coverage: `2` variants x current configured problem sizes

## Key Measurements
- All `2` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=262144`, the fastest median GPU time came from `variant=noop` at `0.009104 ms`. Median GB/s: `0.000`. Median throughput: `0.000`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [dispatch basics best dispatch](./results/tables/dispatch_basics_best_dispatch.csv)
- [dispatch basics best dispatch by device](./results/tables/dispatch_basics_best_dispatch_by_device.csv)
- [dispatch basics device chart index](./results/tables/dispatch_basics_device_chart_index.csv)
- [dispatch basics device summary](./results/tables/dispatch_basics_device_summary.csv)
- [dispatch basics gpu ms pivot](./results/tables/dispatch_basics_gpu_ms_pivot.csv)
- [dispatch basics multi run summary](./results/tables/dispatch_basics_multi_run_summary.csv)
- [benchmark summary](./results/charts/benchmark_summary.png)
- [dispatch basics cross device comparison](./results/charts/dispatch_basics_cross_device_comparison.png)
- [dispatch basics nvidia geforce rtx 2080 super 2480242688 1 4 325 time throughput](./results/charts/dispatch_basics_nvidia_geforce_rtx_2080_super_2480242688_1_4_325_time_throughput.png)
- [dispatch basics operation ratio by device](./results/charts/dispatch_basics_operation_ratio_by_device.png)
- [dispatch basics peak throughput](./results/charts/dispatch_basics_peak_throughput.png)
- [dispatch basics ratio summary](./results/charts/dispatch_basics_ratio_summary.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `52322.50%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
