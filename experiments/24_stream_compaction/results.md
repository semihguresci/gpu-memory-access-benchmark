# Experiment 24 Results: Stream Compaction

## Run Status
- Benchmark status: not yet refreshed for this experiment after implementation
- Test status: pending targeted build/test verification for Experiment 24
- GPU timestamps: required by the implementation
- Variants: `global_atomic_append` and `three_stage`
- Sweep: valid ratios `5, 25, 50, 75, 95` over one fixed logical work size chosen from `--size`

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [stream compaction summary](./results/tables/stream_compaction_summary.csv)
- [stream compaction relative](./results/tables/stream_compaction_relative.csv)
- [stream compaction stability](./results/tables/stream_compaction_stability.csv)
- [stream compaction effective gbps](./results/charts/stream_compaction_effective_gbps.png)
- [stream compaction median gpu ms](./results/charts/stream_compaction_median_gpu_ms.png)
- [stream compaction speedup vs atomic](./results/charts/stream_compaction_speedup_vs_atomic.png)
- [stream compaction stability ratio](./results/charts/stream_compaction_stability_ratio.png)

## Measurement Intent
- `gpu_ms` reports total pipeline runtime for the selected variant.
- `throughput` is normalized as logical input elements processed per second.
- `gbps` is based on estimated useful bytes touched by each implementation, not padded storage size.

## Interpretation Notes
- `three_stage` preserves stable output ordering and validates exact compacted output plus intermediate scan buffers.
- `global_atomic_append` is intentionally order-unstable and validates exact output count plus unordered output contents.
- Cross-variant comparisons should focus on total runtime by valid ratio first; `gbps` is secondary and experiment-local.

## Limitations
- This experiment uses host-visible storage buffers, so absolute values may differ from device-local production paths.
- The current implementation keeps the block-scan stage within one workgroup, which caps the logical problem size.
- Derived tables and charts will remain empty until collection and analysis scripts are run on fresh benchmark data.
