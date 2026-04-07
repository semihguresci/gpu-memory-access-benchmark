# Experiment 03 Results: Memory Copy Baseline

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`75/75` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:48:13Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_134813Z_full_refresh_20260405.json)
- Sweep coverage: `3` variants x current configured problem sizes

## Key Measurements
- All `3` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=4194304`, the fastest median GPU time came from `variant=write_only` at `0.047264 ms`. Median GB/s: `354.968`. Median throughput: `88742044685.173`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [memory copy baseline multi run summary](./results/tables/memory_copy_baseline_multi_run_summary.csv)
- [memory copy baseline runs index](./results/tables/memory_copy_baseline_runs_index.csv)
- [memory copy baseline status overview](./results/tables/memory_copy_baseline_status_overview.csv)
- [memory copy baseline summary](./results/tables/memory_copy_baseline_summary.csv)
- [benchmark summary](./results/charts/benchmark_summary.png)
- [memory copy baseline gbps vs size](./results/charts/memory_copy_baseline_gbps_vs_size.png)
- [memory copy baseline gpu ms vs size](./results/charts/memory_copy_baseline_gpu_ms_vs_size.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `80.43%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
