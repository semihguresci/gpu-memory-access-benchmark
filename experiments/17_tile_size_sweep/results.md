# Experiment 17 Results: Tile Size Sweep

## Run Status
- Benchmark status: latest `full_refresh_20260405` collection completed (`40/40` row correctness pass)
- Test status: `ctest --test-dir build-tests-vs -C Release --output-on-failure` passed `37/37` after integration changes
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --label full_refresh_20260405`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-05T14:01:02Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260405_140102Z_full_refresh_20260405.json)
- Sweep coverage: `8` variants x current configured problem sizes

## Key Measurements
- All `8` benchmark cases are represented in the refreshed export.
- At the largest tested `logical_elements=4194240`, the fastest median GPU time came from `implementation=shared_tiled, tile_size=32, local_size_x=32` at `1.488576 ms`. Median GB/s: `33.811`. Median throughput: `2817618986.199`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [tile size sweep relative](./results/tables/tile_size_sweep_relative.csv)
- [tile size sweep stability](./results/tables/tile_size_sweep_stability.csv)
- [tile size sweep summary](./results/tables/tile_size_sweep_summary.csv)
- [tile size sweep estimated gbps](./results/charts/tile_size_sweep_estimated_gbps.png)
- [tile size sweep median gpu ms](./results/charts/tile_size_sweep_median_gpu_ms.png)
- [tile size sweep speedup vs direct](./results/charts/tile_size_sweep_speedup_vs_direct.png)
- [tile size sweep stability ratio](./results/charts/tile_size_sweep_stability_ratio.png)

## Interpretation
- The fastest and slowest median GPU-time cases in the current focus set are separated by about `110.05%`, so the sweep does show a large spread on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values follow each experiment's own metric definition; compare them within an experiment before comparing them across experiments.
- The refreshed report covers the current sweep only; different sizes, kernels, drivers, or GPUs may shift the ranking.
