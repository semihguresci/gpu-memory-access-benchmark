# Experiment 18 Results: Register Pressure Proxy

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`20/20` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T14:02:14Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_140214Z_full_refresh_20260405.json)
- Sweep coverage: `4` variants x current configured problem sizes

## Key Measurements
- All `4` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=8388608`, the fastest median GPU time came from `variant=temp_4, local_size_x=256` at `2.548768 ms`. Median GB/s: `26.330`. Median throughput: `3291240316.890`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [register pressure proxy relative](./results/tables/register_pressure_proxy_relative.csv)
- [register pressure proxy stability](./results/tables/register_pressure_proxy_stability.csv)
- [register pressure proxy summary](./results/tables/register_pressure_proxy_summary.csv)
- [register pressure proxy median gpu ms](./results/charts/register_pressure_proxy_median_gpu_ms.png)
- [register pressure proxy speedup vs baseline](./results/charts/register_pressure_proxy_speedup_vs_baseline.png)
- [register pressure proxy stability ratio](./results/charts/register_pressure_proxy_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `104.26%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
