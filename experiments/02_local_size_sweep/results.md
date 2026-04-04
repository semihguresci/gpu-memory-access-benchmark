# Experiment 02 Results: Local Size Sweep

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`600/600` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 32M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:09:37Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_140937Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_140937Z_full_refresh_20260329.json)
- Sweep coverage: `2` base variants x `6` local sizes x `10` problem sizes

## Key Measurements
- For `contiguous_write`, the best geometric-mean speedups versus the `local_size_x=64` baseline came from `512` (`1.337x`), `256` (`1.319x`), and `1024` (`1.314x`).
- At the largest tested problem size (`problem_size=8388608`), the fastest `contiguous_write` point was `local_size_x=512` at `0.081056 ms` and `413.966 GB/s`; the `local_size_x=64` baseline landed at `0.136544 ms` and `245.741 GB/s`.
- At that same size, the fastest `noop` point was also `local_size_x=512` at `0.040064 ms` and `837.521 GB/s`.
- The write-over-noop GPU-time ratio worsened from `1.024x` at `local_size_x=32` to `1.347x` at `local_size_x=1024`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Current summary table](./results/tables/local_size_sweep_summary.csv)
- [Best local-size table](./results/tables/local_size_sweep_best_local_size.csv)
- [Local-size ranking table](./results/tables/local_size_sweep_local_size_ranking.csv)
- [Speedup vs `local_size_x=64` chart](./results/charts/local_size_sweep_speedup_vs_ls64.png)
- [Throughput vs local size chart](./results/charts/local_size_sweep_throughput_vs_local_size.png)

## Interpretation
- Larger workgroups are clearly better than the `64` baseline on this GPU for the real write kernel. `512` is now the strongest overall choice across the sweep, with `256` and `1024` still close behind.
- The best `noop` local size is not the same thing as the best latency/throughput tradeoff for every real kernel, but this full-refresh run no longer shows a disagreement between the two at the largest size: both peak at `512` here.

## Limitations
- Results come from one GPU and driver stack.
- The current full-refresh snapshot covers one dispatch per configuration; it does not combine local-size tuning with multi-dispatch batching.
