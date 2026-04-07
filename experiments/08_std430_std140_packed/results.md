# Experiment 08 Results: Std430 Std140 Packed

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`45/45` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:53:40Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135340Z_full_refresh_20260405.json)
- Sweep coverage: `3` variants x current configured problem sizes

## Key Measurements
- All `3` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=524288`, the fastest median GPU time came from `variant=std430` at `21.465152 ms`. Median GB/s: `3.126`. Median throughput: `24425077.446`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [std430 std140 packed layout overview](./results/tables/std430_std140_packed_layout_overview.csv)
- [std430 std140 packed multi run summary](./results/tables/std430_std140_packed_multi_run_summary.csv)
- [std430 std140 packed relative to std430](./results/tables/std430_std140_packed_relative_to_std430.csv)
- [std430 std140 packed runs index](./results/tables/std430_std140_packed_runs_index.csv)
- [std430 std140 packed status overview](./results/tables/std430_std140_packed_status_overview.csv)
- [std430 std140 packed summary](./results/tables/std430_std140_packed_summary.csv)
- [benchmark summary](./results/charts/benchmark_summary.png)
- [std430 std140 packed bandwidth efficiency vs size](./results/charts/std430_std140_packed_bandwidth_efficiency_vs_size.png)
- [std430 std140 packed gbps vs size](./results/charts/std430_std140_packed_gbps_vs_size.png)
- [std430 std140 packed gpu ms vs size](./results/charts/std430_std140_packed_gpu_ms_vs_size.png)
- [std430 std140 packed layout footprint](./results/charts/std430_std140_packed_layout_footprint.png)
- [std430 std140 packed logical gbps vs size](./results/charts/std430_std140_packed_logical_gbps_vs_size.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `44.06%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
