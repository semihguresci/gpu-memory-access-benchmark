# Experiment 12 Results: Gather Access Pattern

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`20/20` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T13:56:56Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_135656Z_full_refresh_20260405.json)
- Sweep coverage: `4` variants x current configured problem sizes

## Key Measurements
- All `4` benchmark cases are represented in the refreshed export.
- At the largest tested `logical_elements=2796202`, the fastest median GPU time came from `variant=block_coherent_32, distribution=block_coherent_32` at `1.737632 ms`. Median GB/s: `19.310`. Median throughput: `1609202638.994`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [gather access pattern relative](./results/tables/gather_access_pattern_relative.csv)
- [gather access pattern stability](./results/tables/gather_access_pattern_stability.csv)
- [gather access pattern summary](./results/tables/gather_access_pattern_summary.csv)
- [gather access pattern median gbps](./results/charts/gather_access_pattern_median_gbps.png)
- [gather access pattern median gpu ms](./results/charts/gather_access_pattern_median_gpu_ms.png)
- [gather access pattern slowdown vs identity](./results/charts/gather_access_pattern_slowdown_vs_identity.png)
- [gather access pattern stability ratio](./results/charts/gather_access_pattern_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `549.34%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
