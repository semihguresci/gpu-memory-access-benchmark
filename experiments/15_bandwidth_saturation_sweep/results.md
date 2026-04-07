# Experiment 15 Results: Bandwidth Saturation Sweep

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`165/165` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:59:17Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135917Z_full_refresh_20260405.json)
- Sweep coverage: `3` variants x current configured problem sizes

## Key Measurements
- All `3` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=33554432`, the fastest median GPU time came from `variant=write_only` at `0.342624 ms`. Median GB/s: `391.735`. Median throughput: `97933688241.337`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [bandwidth saturation multi run summary](./results/tables/bandwidth_saturation_multi_run_summary.csv)
- [bandwidth saturation plateau summary](./results/tables/bandwidth_saturation_plateau_summary.csv)
- [bandwidth saturation runs index](./results/tables/bandwidth_saturation_runs_index.csv)
- [bandwidth saturation status overview](./results/tables/bandwidth_saturation_status_overview.csv)
- [bandwidth saturation summary](./results/tables/bandwidth_saturation_summary.csv)
- [bandwidth saturation gbps vs size](./results/charts/bandwidth_saturation_gbps_vs_size.png)
- [bandwidth saturation gpu ms vs size](./results/charts/bandwidth_saturation_gpu_ms_vs_size.png)
- [benchmark summary](./results/charts/benchmark_summary.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `86.29%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
