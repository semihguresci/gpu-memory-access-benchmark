# Experiment 10 Results: Scalar Type Width Sweep

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`200/200` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 128M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:39:17Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_143917Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_143917Z_full_refresh_20260329.json)
- Sweep coverage: `5` variants x `8` problem sizes

## Key Measurements
- At the largest tested problem size (`problem_size=16777216`), `u8` was fastest at `1.499424 ms`, `22.378 GB/s`, and `4.705x` versus `fp32`.
- `u16` and `fp16_storage` formed the middle tier at `2.982912 ms` / `22.498 GB/s` and `3.005984 ms` / `22.325 GB/s`, both about `2.35x` faster than `fp32`.
- `u32` stayed close to `fp32` (`1.114x` faster), while `fp32` set the baseline at `7.054144 ms` and `19.027 GB/s`.
- Speedup crossover remained selective: `u8` cleared `2x` from the smallest size (`131072`), while `u16` and `fp16_storage` only crossed `2x` at the largest size (`16777216`).
- Stability was best on average for `fp32` (`avg p95/median=1.007x`), while `u32` had the weakest average stability (`1.148x`) and `u8` still showed the largest tail at the biggest size (`p95/median=1.559x`).

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Summary table](./results/tables/scalar_type_width_sweep_summary.csv)
- [Largest-size comparison table](./results/tables/scalar_type_width_sweep_largest_size_comparison.csv)
- [Speedup crossover table](./results/tables/scalar_type_width_sweep_speedup_crossover.csv)
- [Stability overview table](./results/tables/scalar_type_width_sweep_stability_overview.csv)
- [Speedup vs `fp32` chart](./results/charts/scalar_type_width_sweep_speedup_vs_fp32.png)
- [Error-metrics chart](./results/charts/scalar_type_width_sweep_error_metrics.png)

## Interpretation
- The full-refresh dataset still supports the core hypothesis: narrower storage wins because the kernel is largely bandwidth-driven.
- `u8` is the fastest path, while `u16` and `fp16_storage` remain the more balanced middle ground when numerical constraints allow them.

## Limitations
- Results come from one GPU and driver stack.
- This experiment does not include cross-device validation or image-quality/perceptual error analysis beyond the recorded absolute-error metrics.
