# Experiment 39 Results: Tiled Light Assignment

## Run Status
- Benchmark status: latest collection completed (`180/180` row correctness pass)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `5` timed iterations, `2` warmup iterations; current sweep keeps `tile_count_x=16`, `tile_count_y=16`, and `local_size_x=256`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T18:03:40Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_180340Z_2.json)
- Sweep coverage: `6` variants x `6` problem sizes across `2` light distributions

## Key Measurements
- All `36` benchmark cases are represented in the current export.
- At the largest tested `light_count=8388608`, the fastest median GPU time came from `variant=tile_parallel_shared_center_clustered` at `21.639392 ms`. Median GB/s: `1587.833`. Median throughput: `387654514.508`.
- At the same size, `tile_parallel_shared_uniform_lights` stayed close at `21.752032 ms`, while `light_atomic` ranged from `980.314752 ms` to `1162.725760 ms` and `tile_serial` ranged from `1819.016384 ms` to `1819.116608 ms`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [tiled light assignment relative](./results/tables/tiled_light_assignment_relative.csv)
- [tiled light assignment stability](./results/tables/tiled_light_assignment_stability.csv)
- [tiled light assignment summary](./results/tables/tiled_light_assignment_summary.csv)
- [tiled light assignment median gpu ms](./results/charts/tiled_light_assignment_ms.png)
- [tiled light assignment relative ratios](./results/charts/tiled_light_assignment_relative.png)
- [tiled light assignment stability ratio](./results/charts/tiled_light_assignment_stability.png)
- [tiled light assignment throughput](./results/charts/tiled_light_assignment_throughput.png)

## Interpretation
- In the current sweep, the shared-memory tile-parallel path stayed well ahead of both comparison strategies for both light distributions.
- `light_atomic` consistently beat `tile_serial` on median GPU time, but neither approach was close to the `tile_parallel_shared_*` variants at higher light counts.

## Limitations
- Results come from one GPU and driver stack.
- The current sweep fixes `tile_count_x=16`, `tile_count_y=16`, and `local_size_x=256`; different tile geometry or workgroup sizing may change the ranking.
- The derived GB/s metric uses per-variant estimated global traffic, so compare it cautiously across assignment strategies and prefer median GPU time for the main ranking.
