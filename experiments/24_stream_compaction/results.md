# Experiment 24 Results: Stream Compaction

## Run Status
- Benchmark status: latest `20260428_195140Z_2` collection completed (`50/50` row correctness pass across `10` cases)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T19:51:40Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_195140Z_2.json)
- Sweep coverage: `2` implementations x `5` valid ratios at `786432` logical elements

## Key Measurements
- All `10` benchmark cases report `correctness_pass_rate=1.000000`.
- The fastest median GPU time was `implementation=three_stage, valid_ratio_percent=5` at `0.482784 ms`. Median GB/s: `19.949460`. Median throughput: `1628952077.947902`.
- `three_stage` stayed faster than `global_atomic_append` at every tested valid ratio, with speedups from `1.691408x` (`95%` valid) to `2.333400x` (`5%` valid).

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [stream compaction summary](./results/tables/stream_compaction_summary.csv)
- [stream compaction relative](./results/tables/stream_compaction_relative.csv)
- [stream compaction stability](./results/tables/stream_compaction_stability.csv)
- [stream compaction effective gbps](./results/charts/stream_compaction_effective_gbps.png)
- [stream compaction median gpu ms](./results/charts/stream_compaction_median_gpu_ms.png)
- [stream compaction speedup vs atomic](./results/charts/stream_compaction_speedup_vs_atomic.png)
- [stream compaction stability ratio](./results/charts/stream_compaction_stability_ratio.png)

## Interpretation
- In this sweep, the staged path kept a clear runtime advantage while also keeping stable output ordering.
- The advantage narrows as the valid ratio rises, but it remains positive across the full `5%` to `95%` range on this GPU.

## Limitations
- Results come from one GPU/driver stack and one logical size (`786432` elements).
- Cross-variant comparisons keep the experiment's current validation semantics: `three_stage` preserves stable ordering, while `global_atomic_append` validates unordered compaction.
- Reported GB/s values use this experiment's estimated traffic model and are best compared within this experiment.
