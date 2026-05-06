# Experiment 28 Results: Device-Local vs Host-Visible Heap Placement

## Run Status
- Benchmark status: latest `20260429_182842Z` collection completed (`10/10` row correctness pass across `2` benchmark cases)
- Test status: not run in this refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 1G`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-29T18:28:42Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260429_182842Z.json)
- Sweep coverage: `2` variants x current configured problem size (`67108864` logical elements)

## Key Measurements
- All `2` benchmark cases are represented in the refreshed export.
- On the exported full GPU path (`gpu_time_scope=full_gpu_path`), `host_visible_direct` was faster at `27.899424 ms` median versus `47.425344 ms` for `device_local_staged`. That is a `1.699868x` slowdown for the staged device-local path on the current GPU-path metric.
- Median full GPU-path GB/s followed the same ranking: `19.243082` for `host_visible_direct` versus `11.320338` for `device_local_staged`, or `0.588281x` of the host-visible baseline.
- The broader harness wall-clock metric was nearly identical in this run: `11796.678800 ms` for `host_visible_direct` versus `11793.941000 ms` for `device_local_staged`, a `0.999768x` ratio. That wall-clock scope includes CPU-side buffer filling and validation and is not the headline placement metric.
- The staged device-local path carried explicit transfer costs in the median samples: `upload_ms=23.304224` and `readback_ms=22.733472`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [heap placement summary](./results/tables/device_local_vs_host_visible_heap_placement_summary.csv)
- [heap placement relative](./results/tables/device_local_vs_host_visible_heap_placement_relative.csv)
- [heap placement stability](./results/tables/device_local_vs_host_visible_heap_placement_stability.csv)
- [heap placement full gpu path ms](./results/charts/heap_placement_gpu_path_ms.png)
- [heap placement harness wall-clock ms](./results/charts/heap_placement_harness_wall_clock_ms.png)
- [heap placement full gpu path gbps](./results/charts/heap_placement_gpu_path_gbps.png)

## Interpretation
- On this GPU and with the current implementation, host-visible direct placement wins on the experiment's exported full GPU-path metric.
- The near-equal harness wall-clock times do not show that the placement difference disappears. That broader metric is dominated by CPU-side fill and validation work, so it should be treated as harness context rather than the primary placement comparison.

## Limitations
- Results come from one GPU and driver stack.
- The current sweep covers one configured problem size only, so it does not show where the placement tradeoff might change across smaller or larger working sets.
- The exported `gpu_ms` and `gbps` values represent the experiment's full GPU path, while `end_to_end_ms` is a broader harness wall-clock measurement that includes CPU-side setup and oracle work.
