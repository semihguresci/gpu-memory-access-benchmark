# Experiment 06 Results: Aos Vs Soa

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`10/10` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:50:31Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135031Z_full_refresh_20260405.json)
- Sweep coverage: `2` variants x current configured problem sizes

## Key Measurements
- All `2` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=1000000`, the fastest median GPU time came from `variant=soa` at `2.465888 ms`. Median GB/s: `19.466`. Median throughput: `405533422.443`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [aos vs soa multi run summary](./results/tables/aos_vs_soa_multi_run_summary.csv)
- [aos vs soa runs index](./results/tables/aos_vs_soa_runs_index.csv)
- [aos vs soa status overview](./results/tables/aos_vs_soa_status_overview.csv)
- [aos vs soa summary](./results/tables/aos_vs_soa_summary.csv)
- [aos vs soa gbps vs size](./results/charts/aos_vs_soa_gbps_vs_size.png)
- [aos vs soa gpu ms vs size](./results/charts/aos_vs_soa_gpu_ms_vs_size.png)
- [benchmark summary](./results/charts/benchmark_summary.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `2751.14%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
