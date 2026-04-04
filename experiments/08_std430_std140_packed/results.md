# Experiment 08 Results: std430 vs std140 vs Packed

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`60/60` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 128M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:34:02Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_143402Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_143402Z_full_refresh_20260329.json)
- Sweep coverage: `3` variants x `4` problem sizes

## Key Measurements
- At the largest tested problem size (`problem_size=1048576`), `std430` measured `46.837856 ms`, `3.582 GB/s` storage bandwidth, and `2.866 GB/s` logical payload bandwidth.
- `std140` measured `53.234048 ms` and `4.412 GB/s` storage bandwidth, but only `2.521 GB/s` logical payload bandwidth; it was `1.137x` slower than `std430`.
- `packed` eliminated alignment waste entirely, but still measured `61.083744 ms` and `2.197 GB/s` logical payload bandwidth, or `1.304x` slower than `std430`.
- Layout overhead at the largest size was `75%` for `std140`, `25%` for `std430`, and `0%` for `packed` according to [the layout overview](./results/tables/std430_std140_packed_layout_overview.csv).

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Current summary table](./results/tables/std430_std140_packed_summary.csv)
- [Layout overview table](./results/tables/std430_std140_packed_layout_overview.csv)
- [Relative-to-`std430` table](./results/tables/std430_std140_packed_relative_to_std430.csv)
- [GPU time chart](./results/charts/std430_std140_packed_gpu_ms_vs_size.png)
- [Logical GB/s chart](./results/charts/std430_std140_packed_logical_gbps_vs_size.png)
- [Layout footprint chart](./results/charts/std430_std140_packed_layout_footprint.png)

## Interpretation
- `std430` remains the best latency choice in the full-refresh dataset.
- `std140` can show a higher storage-bandwidth number because the metric counts padded bytes moved. Logical payload bandwidth and elapsed time are still the better measures of useful work here.
- `packed` removes padding completely, but on this GPU it still loses in elapsed time for this kernel.

## Limitations
- Results come from one GPU and driver stack.
- The benchmark measures one kernel and one data layout contract; different shader code or device alignment rules could change the relative ordering.
