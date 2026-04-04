# Experiment 14 Results: Read Reuse Cache Locality

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`25/25` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 32M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:43:15Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_144315Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_144315Z_full_refresh_20260329.json)
- Sweep coverage: `5` variants x `1` logical size

## Key Measurements
- All variants kept the same logical workload at `logical_elements=8388608` with `source_unique_elements=4194304` and a fixed `source_reuse_ratio=2.0`.
- `reuse_distance_32` posted the best median at `3.902752 ms` and `25.793 GB/s`, but it was effectively tied with `reuse_distance_1` (`3.905056 ms`), `reuse_distance_256` (`3.910720 ms`), and `reuse_distance_4096` (`3.909792 ms`).
- `reuse_distance_full_span` was still the only clear slowdown at `5.171296 ms` and `19.466 GB/s`, making every bounded-reuse variant about `1.32x` faster on median.
- The largest bounded variant, `reuse_distance_4096`, did not lose median performance versus the smaller reuse blocks; its `p95/median` landed at `1.249x`.
- `reuse_distance_1` was the steadiest measured variant at `p95/median=1.190x`, while `reuse_distance_full_span` remained slower but no longer showed the outsized tail seen in the previous refresh (`p95/median=1.167x`).

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/read_reuse_cache_locality_summary.csv)
- [Relative table](./results/tables/read_reuse_cache_locality_relative.csv)
- [Stability table](./results/tables/read_reuse_cache_locality_stability.csv)
- [Locality table](./results/tables/read_reuse_cache_locality_locality.csv)
- [Median GPU time chart](./results/charts/read_reuse_cache_locality_median_gpu_ms.png)
- [Speedup chart](./results/charts/read_reuse_cache_locality_speedup_vs_full_span.png)
- [Stability chart](./results/charts/read_reuse_cache_locality_stability_ratio.png)

## Interpretation
- On this GPU and workload size, the benchmark still shows no obvious median decay as pair reuse is deferred from `1` read out to `4096` reads. The only pronounced penalty appears when the second touch is pushed to a full-span replay across the entire half-sized source set.
- That result is consistent with bounded reuse staying inside a favorable cache or locality regime, but the timing data alone still does not prove a specific cache capacity boundary.

## Limitations
- Results come from one GPU and driver stack.
- The run uses one logical problem size and no hardware counters, so the locality conclusion is timing-based only.
- Although the tails are much cleaner in this refresh, the conclusion should still be treated as directional until it is repeated on additional GPUs or with larger sample counts.
