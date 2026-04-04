# Experiment 13 Results: Scatter Access Pattern

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`15/15` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 32M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:42:12Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_144212Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_144212Z_full_refresh_20260329.json)
- Sweep coverage: `3` variants x `1` logical size

## Key Measurements
- `unique_permutation` set the baseline at `48.199520 ms` and `2.088 GB/s` for `logical_elements=8388608`.
- `random_collision_x4` stayed effectively tied and was slightly faster in this run at `47.079488 ms` and `2.138 GB/s`, a `2.32%` improvement versus the baseline median.
- `clustered_hotset_32` measured `108.315712 ms` and `0.929 GB/s`, or `2.247x` slower than `unique_permutation`.
- The contention metadata still explains the split: `random_collision_x4` used `active_target_count=2097152` with `max_expected_counter=4`, while `clustered_hotset_32` compressed traffic to `active_target_count=1048576` with `max_expected_counter=8`.
- All three variants were fairly stable in this refresh: `unique_permutation` was steadiest at `p95/median=1.007x`, `clustered_hotset_32` stayed near `1.013x`, and `random_collision_x4` reached `1.020x`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/scatter_access_pattern_summary.csv)
- [Relative table](./results/tables/scatter_access_pattern_relative.csv)
- [Stability table](./results/tables/scatter_access_pattern_stability.csv)
- [Contention table](./results/tables/scatter_access_pattern_contention.csv)
- [Slowdown chart](./results/charts/scatter_access_pattern_slowdown_vs_unique_permutation.png)
- [Stability chart](./results/charts/scatter_access_pattern_stability_ratio.png)

## Interpretation
- Spread-out collisions remained effectively tied with the unique-target baseline in the full-refresh run. Localized contention is still the dominant penalty source here, not mere target indirection.
- The `GB/s` column in this experiment is a logical traffic proxy from the benchmark contract, not a hardware-counter measurement of physical atomic bandwidth. The timing comparison remains the primary result.

## Limitations
- Results come from one GPU and driver stack.
- No hardware counters or cache/atomic throughput counters were captured.
