# Experiment 07 Results: Aosoa Blocked Layout

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`60/60` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:53:26Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135326Z_full_refresh_20260405.json)
- Sweep coverage: `6` variants x current configured problem sizes

## Key Measurements
- All `6` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=2000000`, the fastest median GPU time came from `variant=soa` at `7.540736 ms`. Median GB/s: `21.218`. Median throughput: `265226099.946`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [aosoa blocked layout multi run summary](./results/tables/aosoa_blocked_layout_multi_run_summary.csv)
- [aosoa blocked layout runs index](./results/tables/aosoa_blocked_layout_runs_index.csv)
- [aosoa blocked layout status overview](./results/tables/aosoa_blocked_layout_status_overview.csv)
- [aosoa blocked layout summary](./results/tables/aosoa_blocked_layout_summary.csv)
- [aosoa blocked layout gbps vs size](./results/charts/aosoa_blocked_layout_gbps_vs_size.png)
- [aosoa blocked layout gpu ms vs size](./results/charts/aosoa_blocked_layout_gpu_ms_vs_size.png)
- [benchmark summary](./results/charts/benchmark_summary.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `1414.79%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
