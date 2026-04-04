# Experiment 11 Results: Coalesced vs Strided Access

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`35/35` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 128M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:40:38Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_144038Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_144038Z_full_refresh_20260329.json)
- Sweep coverage: `7` stride variants x `1` logical size

## Key Measurements
- The coalesced baseline `stride_1` measured `0.199328 ms` and `21.042 GB/s`.
- `stride_2` was the worst case at `17.756576 ms` and `0.236 GB/s`, or `89.082x` slower than `stride_1`.
- The fastest strided case was `stride_16` at `2.786240 ms` and `1.505 GB/s`, which still remained `13.978x` slower than the coalesced baseline.
- Stability was strongest for `stride_1` (`p95/median=1.002x`) and weakest for `stride_32` (`p95/median=1.455x`).

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/coalesced_vs_strided_summary.csv)
- [Relative table](./results/tables/coalesced_vs_strided_relative.csv)
- [Stability table](./results/tables/coalesced_vs_strided_stability.csv)
- [Slowdown chart](./results/charts/coalesced_vs_strided_slowdown_vs_stride_1.png)
- [Stability chart](./results/charts/coalesced_vs_strided_stability_ratio.png)

## Interpretation
- The full-refresh run preserves the same non-monotonic shape: the first loss of coalescing hurts the most, and larger strides settle into a slower plateau rather than degrading linearly forever.
- A simple rule like "bigger stride means proportionally bigger slowdown" still does not fit this dataset. Transaction behavior is clearly more complex than that.

## Limitations
- Results come from one GPU and driver stack.
- No hardware counters were captured, so the inferred transaction behavior is based on timing only.
