# Experiment 07 Results

## Run Snapshot
- Status: smoke run completed (`18/18` correctness pass)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2479554560`)
- Config: `--iterations 1 --warmup 0 --size 256M`
- Latest run: `runs/nvidia_geforce_rtx_2080_super/20260315_141304Z.json`

## Key Measurements (Smoke)
- `problem_size=4000000`:
  - `soa`: `15.030880 ms`, `21.2895 GB/s`
  - `aosoa_b32`: `15.280352 ms`, `20.9419 GB/s`
  - `aosoa_b16`: `16.865344 ms`, `18.9738 GB/s`
  - `aosoa_b8`: `24.027008 ms`, `13.3183 GB/s`
  - `aosoa_b4`: `37.566496 ms`, `8.5182 GB/s`
  - `aos`: `216.156160 ms`, `2.3687 GB/s`

## Graphics
![Dispatch Time](./results/charts/aosoa_blocked_layout_gpu_ms_vs_size.png)

![Effective Bandwidth](./results/charts/aosoa_blocked_layout_gbps_vs_size.png)

![Summary](./results/charts/benchmark_summary.png)

## Data Links
- [Summary table](./results/tables/aosoa_blocked_layout_summary.csv)
- [Status overview](./results/tables/aosoa_blocked_layout_status_overview.csv)
- [Runs index](./results/tables/aosoa_blocked_layout_runs_index.csv)
- [Multi-run summary](./results/tables/aosoa_blocked_layout_multi_run_summary.csv)
- [Raw benchmark export](./results/tables/benchmark_results.json)

## Interpretation and Limits
- This is a smoke dataset only (`timed_iterations=1`), so variance and tail latency are not characterized.
- Correctness is green for all variants on this run, which validates indexing/packing paths for first-pass implementation.
- Initial throughput ordering shows `soa` leading, with blocked AoSoA variants clustered behind and `aos` far slower for this access pattern.
