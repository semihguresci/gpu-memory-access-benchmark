# Experiment 15 Results: Bandwidth Saturation Sweep

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`225/225` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 512M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:53:20Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_145320Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_145320Z_full_refresh_20260329.json)
- Sweep coverage: `3` variants x `15` size points (`1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 MiB`)

## Key Measurements
- `read_only` reached its best median at `512 MiB`: `0.589856 ms` and `910.173 GB/s`. The current plateau heuristic marked `96 MiB` as the sustained-region start, with a sustained median of `899.028 GB/s` over the `96-512 MiB` window.
- `read_write_copy` also peaked at `512 MiB`: `2.534176 ms` and `423.705 GB/s`. Its plateau heuristic triggered much earlier, at `24 MiB`, with a sustained median of `419.305 GB/s`.
- `write_only` was strongest at `96 MiB`: `0.260544 ms` and `386.358 GB/s`. The plateau heuristic marked `64 MiB` as the sustained-region start, with a sustained median of `381.578 GB/s`.
- Unlike the earlier anomaly-heavy run, the current plateau heuristic triggered for all three variants in [bandwidth_saturation_plateau_summary.csv](./results/tables/bandwidth_saturation_plateau_summary.csv).

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Current summary table](./results/tables/bandwidth_saturation_summary.csv)
- [Plateau summary table](./results/tables/bandwidth_saturation_plateau_summary.csv)
- [Status overview table](./results/tables/bandwidth_saturation_status_overview.csv)
- [Multi-run summary table](./results/tables/bandwidth_saturation_multi_run_summary.csv)
- [GPU time chart](./results/charts/bandwidth_saturation_gpu_ms_vs_size.png)
- [GB/s chart](./results/charts/bandwidth_saturation_gbps_vs_size.png)
- [Two-panel benchmark summary](./results/charts/benchmark_summary.png)

## Interpretation
- This full-refresh dataset now shows the intended overhead-amortization pattern. All three variants flatten into sustained bandwidth bands over larger transfer sizes instead of collapsing after their early best points.
- `read_only` continues improving deepest into the sweep, while `write_only` and `read_write_copy` settle earlier into narrower plateaus. That makes Experiment 15 much closer to the architecture-aware saturation reference the plan intended.

## Limitations
- Results come from one GPU and driver stack.
- Plateau detection is still heuristic-driven and timing-based only; no hardware counters were collected to confirm the exact saturation mechanism.
- The current conclusions should still be rechecked on additional GPUs before they are treated as cross-architecture truths.
