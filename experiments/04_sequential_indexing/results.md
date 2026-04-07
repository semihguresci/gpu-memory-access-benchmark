# Experiment 04 Results: Sequential Indexing

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`520/520` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:48:46Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_134846Z_full_refresh_20260405.json)
- Sweep coverage: `1` variants x current configured problem sizes

## Key Measurements
- All `1` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=4194304`, the fastest median GPU time came from `variant=sequential_read_write` at `0.084256 ms`. Median GB/s: `398.244`. Median throughput: `49780478541.588`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [sequential indexing multi run summary](./results/tables/sequential_indexing_multi_run_summary.csv)
- [sequential indexing runs index](./results/tables/sequential_indexing_runs_index.csv)
- [sequential indexing status overview](./results/tables/sequential_indexing_status_overview.csv)
- [sequential indexing summary](./results/tables/sequential_indexing_summary.csv)
- [benchmark summary](./results/charts/benchmark_summary.png)
- [sequential indexing gbps vs size](./results/charts/sequential_indexing_gbps_vs_size.png)
- [sequential indexing gpu ms vs size](./results/charts/sequential_indexing_gpu_ms_vs_size.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `98947.74%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
