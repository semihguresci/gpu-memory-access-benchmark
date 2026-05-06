# Experiment 32 Results: Subgroup Stream Compaction Variants

## Run Status
- Benchmark status: latest `20260429_180654Z` collection completed (`50/50` row correctness pass across `10` benchmark cases)
- Test status: not run in this refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 1G`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-29T18:06:54Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260429_180654Z_2.json)
- Sweep coverage: `2` variants x `5` valid-ratio settings at the current configured problem size (`67108864` elements)

## Key Measurements
- All `10` benchmark cases are represented in the refreshed export.
- The fastest median GPU time came from `shared_atomic_block_ratio_5` at `17.226112 ms`. Median GB/s: `16.423157`. Median throughput: `3895763826.451`.
- The stable subgroup-ballot path did not win any valid-ratio point in this run. Its closest result was at `25%` valid input, where `subgroup_ballot_ratio_25` measured `23.518080 ms` versus `22.274752 ms` for `shared_atomic_block_ratio_25`, or `0.947133x` of the shared baseline throughput.
- The largest subgroup regression appeared at `75%` valid input: `subgroup_ballot_ratio_75` measured `26.919840 ms` versus `22.024416 ms`, only `0.818148x` of the shared baseline throughput.
- At the other valid ratios, the subgroup path remained behind: `0.879373x` at `5%`, `0.862842x` at `50%`, and `0.851873x` at `95%`.
- The largest p95-to-median GPU-time ratio in the current stability table is `1.381869` for `shared_atomic_block_ratio_5`, while the subgroup path peaks at `1.207900` for `subgroup_ballot_ratio_5`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [subgroup compaction summary](./results/tables/subgroup_stream_compaction_variants_summary.csv)
- [subgroup compaction relative](./results/tables/subgroup_stream_compaction_variants_relative.csv)
- [subgroup compaction stability](./results/tables/subgroup_stream_compaction_variants_stability.csv)
- [subgroup compaction gpu ms](./results/charts/subgroup_compaction_gpu_ms.png)
- [subgroup compaction speedup](./results/charts/subgroup_compaction_speedup.png)
- [subgroup compaction stability](./results/charts/subgroup_compaction_stability.png)

## Interpretation
- On this GPU and with the current block-local kernels, the shared-atomic block compaction path is consistently faster than the stable subgroup-ballot path across the full valid-ratio sweep.
- The subgroup path still preserves stable ordering, but that extra property is not paying back its cost in the current measurement range.

## Limitations
- Results come from one GPU and driver stack.
- The current sweep covers one configured problem size only, so it does not show whether the ranking changes for smaller or larger compaction workloads.
- Both variants compact independently inside each workgroup-sized block and write per-block counts; this is not a full globally compacted stream like Experiment 24.
- This experiment compares a stable subgroup-ballot implementation against a shared-atomic baseline that does not preserve ordering, so the performance numbers should be read together with that semantic difference.
