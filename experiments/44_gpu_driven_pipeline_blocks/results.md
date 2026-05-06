# Experiment 44 Results: GPU-Driven Pipeline Building Blocks

## Run Status
- Benchmark status: latest `20260428_204146Z_2` collection completed (`60/60` timed rows correctness pass across `12` benchmark cases)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `5` timed iterations, `2` warmup iterations
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T20:41:46Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_204146Z_2.json)
- Sweep coverage: `2` variants x `6` problem sizes (`1048576` to `67108864` elements)

## Key Measurements
- All `12` benchmark cases are represented in the refreshed export, and every summary row reports `correctness_pass_rate=1.0`.
- `staged_three_dispatch` outperformed `fused_single_dispatch` at every tested size; the relative table shows `fused_single_dispatch` at `1.294324x` to `1.482856x` slower than staged.
- At the largest tested `problem_size=67108864`, `staged_three_dispatch` recorded `1037.077984 ms` median GPU time versus `1342.314944 ms` for `fused_single_dispatch`. Median throughput: `64709563.827748` vs `49994872.142316`.
- The widest p95-to-median timing spread in the current sweep was `1.296678` for `fused_single_dispatch` at `problem_size=33554432`.

## Artifact Links
- [Latest collected run JSON](./runs/nvidia_geforce_rtx_2080_super/20260428_204146Z_2.json)
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [gpu driven pipeline blocks relative](./results/tables/gpu_driven_pipeline_blocks_relative.csv)
- [gpu driven pipeline blocks stability](./results/tables/gpu_driven_pipeline_blocks_stability.csv)
- [gpu driven pipeline blocks summary](./results/tables/gpu_driven_pipeline_blocks_summary.csv)
- [gpu driven pipeline blocks median gpu ms](./results/charts/gpu_driven_pipeline_blocks_ms.png)
- [gpu driven pipeline blocks relative comparison](./results/charts/gpu_driven_pipeline_blocks_relative.png)
- [gpu driven pipeline blocks stability ratio](./results/charts/gpu_driven_pipeline_blocks_stability.png)
- [gpu driven pipeline blocks throughput](./results/charts/gpu_driven_pipeline_blocks_throughput.png)

## Interpretation
- On this RTX 2080 SUPER run, the staged path remained faster even though it uses `3` dispatches per iteration versus `1` for the fused path.
- The advantage narrowed at larger problem sizes but did not reverse anywhere in the refreshed sweep.

## Limitations
- Results come from one GPU and driver stack.
- The current report covers the configured sweep only; different visibility thresholds, bucket counts, local sizes, or devices may change the ranking.
- Reported throughput and GB/s values follow this experiment's metric definitions and are most reliable for within-experiment comparisons.
