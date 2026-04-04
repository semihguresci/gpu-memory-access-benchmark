# Experiment 16 Results: Shared Memory Tiling

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`40/40` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 32M --label full_refresh_20260329`
- Workload: fixed `logical_elements=8388352`, `local_size_x=256`, `2` implementations (`direct_global`, `shared_tiled`) x `4` reuse radii (`1, 4, 8, 16`)
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:55:24Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_145524Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_145524Z_full_refresh_20260329.json)

## Key Measurements
- All `8` benchmark cases reported `correctness_pass_rate=1.0`.
- Radius `1`: `direct_global` beat `shared_tiled` at `3.025952 ms` vs `3.103488 ms`, so tiling was `0.975x` as fast on median and about `2.56%` slower.
- Radius `4`: `direct_global` again led at `3.024960 ms` vs `3.111296 ms`, leaving tiling at `0.972x` of direct throughput and about `2.85%` slower on median.
- Radius `8`: `direct_global` remained slightly ahead at `3.025824 ms` vs `3.109280 ms`, with tiling at `0.973x` of direct throughput.
- Radius `16`: the two paths were nearly tied, but `direct_global` still held the lead at `2.998336 ms` vs `3.017824 ms`, making tiling `0.994x` as fast and about `0.65%` slower.
- Tail behavior was much tighter than in the earlier moderate run: `p95/median` stayed between about `1.09x` and `1.37x` across all cases.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/shared_memory_tiling_summary.csv)
- [Relative table](./results/tables/shared_memory_tiling_relative.csv)
- [Stability table](./results/tables/shared_memory_tiling_stability.csv)
- [Median GPU time chart](./results/charts/shared_memory_tiling_median_gpu_ms.png)
- [Estimated GB/s chart](./results/charts/shared_memory_tiling_estimated_gbps.png)
- [Speedup chart](./results/charts/shared_memory_tiling_speedup_vs_direct.png)
- [Stability chart](./results/charts/shared_memory_tiling_stability_ratio.png)

## Interpretation
- This full-refresh dataset no longer shows a reuse-radius crossover where shared-memory staging clearly wins. On this GPU and with the current first-draft kernel, `direct_global` is marginally faster at every measured radius.
- That pattern suggests the fixed staging costs in the tiled kernel, including extra shared-memory traffic and one barrier per workgroup, are not yet being repaid by reuse on this workload shape. The gap is small enough that the result should still be treated as directional, not final.

## Limitations
- Results come from one GPU and driver stack.
- Reported `GB/s` values are derived from estimated global read and write bytes, not hardware performance counters.
- The experiment keeps `local_size_x=256` fixed and uses one logical problem size; tile-size sensitivity is intentionally deferred to Experiment 17.
