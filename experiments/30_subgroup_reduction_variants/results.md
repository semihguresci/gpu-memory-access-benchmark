# Experiment 30 Results: Subgroup Reduction Variants

## Run Status
- Benchmark status: latest `20260428_185933Z` collection completed (`40/40` row correctness pass across `8` benchmark cases)
- Test status: not run in this refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 1G`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T18:59:33Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_185933Z_2.json)
- Sweep coverage: `2` variants x `4` problem sizes (`655360` through `41943040` elements)

## Key Measurements
- All `8` benchmark cases are represented in the refreshed export.
- At the largest tested `problem_size=41943040`, `shared_tree` recorded the fastest median GPU time at `13.318848 ms`. Median GB/s: `12.596597`. Median throughput: `3149149235.730`.
- `subgroup_hybrid` only led at the smallest tested size, where it measured `0.231392 ms` versus `0.233792 ms` for `shared_tree`, a `1.010372x` speedup.
- At the other three tested sizes, `subgroup_hybrid` was slightly slower than `shared_tree`, with speedup ratios of `0.994563`, `0.996152`, and `0.999136`.
- The largest p95-to-median GPU-time ratio in the current stability table is `1.028347` for `subgroup_hybrid` at `problem_size=41943040`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [subgroup reduction summary](./results/tables/subgroup_reduction_variants_summary.csv)
- [subgroup reduction relative](./results/tables/subgroup_reduction_variants_relative.csv)
- [subgroup reduction stability](./results/tables/subgroup_reduction_variants_stability.csv)
- [subgroup reduction gpu ms](./results/charts/subgroup_reduction_gpu_ms.png)
- [subgroup reduction speedup](./results/charts/subgroup_reduction_speedup.png)
- [subgroup reduction stability](./results/charts/subgroup_reduction_stability.png)

## Interpretation
- On this GPU and with the current kernels, subgroup assistance does not produce a consistent reduction win. The two implementations are effectively near-parity once the problem size grows beyond the smallest test case.
- The current data therefore suggests that the shipped `subgroup_hybrid` path is competitive, but not materially better than the shared-tree baseline on the RTX 2080 SUPER.

## Limitations
- Results come from one GPU and driver stack.
- The current sweep covers four configured problem sizes only, so it does not show whether the ranking changes outside this range.
- The measured differences are small for three of the four sizes, so this experiment is better read as "near-parity on this GPU" than as a strong win for either implementation.
