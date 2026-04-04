# Experiment 05 Results: Global ID Mapping Variants

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`1800/1800` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 64M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:18:54Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_141854Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_141854Z_full_refresh_20260329.json)
- Sweep coverage: `3` variants x `15` problem sizes x `8` dispatch counts

## Key Measurements
- At the largest tested problem size (`problem_size=16777216`) and heaviest dispatch point (`dispatch_count=1024`), `offset` measured `330.700768 ms` and `415.599 GB/s`, `grid_stride` measured `330.842848 ms` and `415.421 GB/s`, and `direct` measured `330.855840 ms` and `415.404 GB/s`.
- At the lighter `dispatch_count=4` point, `direct` was fastest at `1.270336 ms` and `422.621 GB/s`, with `grid_stride` at `420.334 GB/s` and `offset` at `414.334 GB/s`.
- Most dispatch counts stayed within roughly `0.3-2.0%` between the best and worst mapping variant; the worst spread was `1.043x` at `dispatch_count=128`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Current summary table](./results/tables/global_id_mapping_variants_summary.csv)
- [Status overview table](./results/tables/global_id_mapping_variants_status_overview.csv)
- [GPU time chart](./results/charts/global_id_mapping_variants_gpu_ms_vs_size.png)
- [GB/s chart](./results/charts/global_id_mapping_variants_gbps_vs_size.png)

## Interpretation
- The mapping arithmetic stays a small effect compared with the memory work. All three variants remain tightly clustered through most of the sweep.
- `direct` leads the lighter cases, while `offset` edges ahead at the heaviest dispatch count, but the differences are still too small to dominate design choices on this GPU.

## Limitations
- Results come from one GPU and driver stack.
- The benchmark isolates simple ID remapping overheads; it does not cover more complex per-thread control flow or divergent bounds handling.
