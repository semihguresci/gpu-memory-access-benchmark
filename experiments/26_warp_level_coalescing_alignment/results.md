# Experiment 26 Results: Warp-Level Coalescing Alignment

## Run Status
- Benchmark status: not run yet
- Test status: not run yet
- GPU: pending first measured desktop run
- Config: pending
- Validation layers: pending
- GPU timestamps: pending

## Key Measurements
- No measured data yet.
- Planned sweep: `0`, `4`, `8`, `16`, `32`, `64` bytes.
- Fixed useful work: one `uint32_t` read plus one `uint32_t` write per logical element.
- Validation: deterministic CPU reference using `transform_value(input[base_offset + index], index)`.

## Artifact Links
- Raw benchmark export: `results/tables/benchmark_results.json`
- Summary table: `results/tables/warp_level_coalescing_alignment_summary.csv`
- Relative table: `results/tables/warp_level_coalescing_alignment_relative.csv`
- Stability table: `results/tables/warp_level_coalescing_alignment_stability.csv`
- Footprint table: `results/tables/warp_level_coalescing_alignment_footprint.csv`
- Charts: `results/charts/warp_level_coalescing_alignment_median_gpu_ms.png`, `results/charts/warp_level_coalescing_alignment_median_gbps.png`, `results/charts/warp_level_coalescing_alignment_slowdown_vs_aligned.png`
- Profiler screenshot: pending

## Interpretation
- Pending first benchmark run.
- The intended comparison is alignment-sensitive contiguous access against the aligned baseline, not stride-sensitive divergence.
- Any transaction-boundary explanation should be treated as inference unless a profiler confirms it.

## Limitations
- No data collected yet.
- The experiment is intentionally scoped to a single access pattern and a fixed arithmetic footprint.
