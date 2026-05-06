# Experiment 27 Results: Cache Thrashing, Random vs Sequential

## Run Status
- Benchmark status: latest `20260428_182847Z` collection completed (`15/15` row correctness pass across `3` benchmark cases)
- Test status: not run in this refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 1G`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T18:28:47Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_182847Z_3.json)
- Sweep coverage: `3` variants x current configured problem size (`89478400` logical elements, `357913600` bytes per buffer)

## Key Measurements
- All `3` benchmark cases are represented in the refreshed export.
- The fastest median GPU time came from `variant=sequential` at `58.589824 ms`. Median estimated GB/s: `18.326`. Median throughput: `1527200354.792`.
- `block_shuffled` stayed close to the good-path baseline at `59.082624 ms`, which is a `1.008411x` slowdown versus `sequential`.
- `random` dropped to `693.667968 ms`, `1.547918` median estimated GB/s, and `128993126.579` throughput. That is an `11.839393x` slowdown and only `0.084464x` of the sequential throughput.
- The largest p95-to-median GPU-time ratio in the current stability table is `1.067617` for `sequential`; the `random` case reached `1.023125`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [cache thrashing summary](./results/tables/cache_thrashing_random_vs_sequential_summary.csv)
- [cache thrashing relative](./results/tables/cache_thrashing_random_vs_sequential_relative.csv)
- [cache thrashing stability](./results/tables/cache_thrashing_random_vs_sequential_stability.csv)
- [cache thrashing median gpu ms](./results/charts/cache_thrashing_random_vs_sequential_median_gpu_ms.png)
- [cache thrashing slowdown vs sequential](./results/charts/cache_thrashing_random_vs_sequential_slowdown_vs_sequential.png)
- [cache thrashing estimated gbps](./results/charts/cache_thrashing_random_vs_sequential_estimated_gbps.png)
- [cache thrashing stability ratio](./results/charts/cache_thrashing_random_vs_sequential_stability_ratio.png)

## Interpretation
- On this GPU, fully random access destroys effective throughput relative to the sequential baseline, while the current `block_shuffled` variant remains close to sequential behavior.
- The current experiment configuration therefore does show the intended separation between healthy locality and deliberate cache defeat.

## Limitations
- Results come from one GPU and driver stack.
- The current sweep covers only one configured working-set size and one `block_size=32`, so it does not yet show how the locality collapse evolves across sizes or block widths.
- Reported GB/s values follow the experiment's estimated traffic model; compare them within this experiment before comparing them across experiments.
