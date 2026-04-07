# Experiment 05 Results: Global Id Mapping Variants

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`1560/1560` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:50:23Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135023Z_full_refresh_20260405.json)
- Sweep coverage: `3` variants x current configured problem sizes

## Key Measurements
- All `3` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=4194304`, the fastest median GPU time came from `variant=offset` at `0.079968 ms`. Median GB/s: `419.598`. Median throughput: `52449779911.965`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [global id mapping variants multi run summary](./results/tables/global_id_mapping_variants_multi_run_summary.csv)
- [global id mapping variants runs index](./results/tables/global_id_mapping_variants_runs_index.csv)
- [global id mapping variants status overview](./results/tables/global_id_mapping_variants_status_overview.csv)
- [global id mapping variants summary](./results/tables/global_id_mapping_variants_summary.csv)
- [benchmark summary](./results/charts/benchmark_summary.png)
- [global id mapping variants gbps vs size](./results/charts/global_id_mapping_variants_gbps_vs_size.png)
- [global id mapping variants gpu ms vs size](./results/charts/global_id_mapping_variants_gpu_ms_vs_size.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `104287.84%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
