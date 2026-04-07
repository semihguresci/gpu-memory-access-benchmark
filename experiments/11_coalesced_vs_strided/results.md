# Experiment 11 Results: Coalesced Vs Strided

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`35/35` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:56:36Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135636Z_full_refresh_20260405.json)
- Sweep coverage: `7` variants x current configured problem sizes

## Key Measurements
- All `7` benchmark cases are represented in the refreshed export.
- At the largest tested `logical_elements=262144`, the fastest median GPU time came from `variant=stride_1` at `0.106400 ms`. Median GB/s: `19.710`. Median throughput: `2463759398.496`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [coalesced vs strided footprint](./results/tables/coalesced_vs_strided_footprint.csv)
- [coalesced vs strided relative](./results/tables/coalesced_vs_strided_relative.csv)
- [coalesced vs strided stability](./results/tables/coalesced_vs_strided_stability.csv)
- [coalesced vs strided summary](./results/tables/coalesced_vs_strided_summary.csv)
- [coalesced vs strided median gbps](./results/charts/coalesced_vs_strided_median_gbps.png)
- [coalesced vs strided median gpu ms](./results/charts/coalesced_vs_strided_median_gpu_ms.png)
- [coalesced vs strided slowdown vs stride 1](./results/charts/coalesced_vs_strided_slowdown_vs_stride_1.png)
- [coalesced vs strided stability ratio](./results/charts/coalesced_vs_strided_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `8691.94%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
