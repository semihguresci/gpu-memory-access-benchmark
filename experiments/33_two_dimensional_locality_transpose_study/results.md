# Experiment 33 Results: 2D Locality and Transpose Study

## Run Status
- Benchmark status: latest `20260428_194914Z` collection completed (`20/20` row correctness pass across `4` benchmark cases)
- Test status: not run in this refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 1G`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T19:49:14Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_194914Z_2.json)
- Sweep coverage: `4` variants at the current configured matrix size (`matrix_dim=11584`, `problem_size=134189056`)

## Key Measurements
- All `4` benchmark cases are represented in the refreshed export.
- The fastest median GPU time came from `row_major_copy` at `62.009504 ms`. Median GB/s: `17.312063`. Median throughput: `2164007891.435`.
- `naive_transpose` was the slowest path at `176.214528 ms`, which is a `2.841734x` slowdown versus `row_major_copy`.
- Both tiled transpose variants materially improved over the naive transpose path: `tiled_transpose` reached `107.690496 ms` (`1.636305x` speedup vs naive), and `tiled_transpose_padded` reached `107.603808 ms` (`1.637624x` speedup vs naive).
- The padded tiled variant was only marginally faster than the unpadded tiled variant in this run, by about `0.086688 ms`.
- The largest p95-to-median GPU-time ratio in the current stability table is `1.065651` for `naive_transpose`.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [transpose summary](./results/tables/two_dimensional_locality_transpose_study_summary.csv)
- [transpose relative](./results/tables/two_dimensional_locality_transpose_study_relative.csv)
- [transpose stability](./results/tables/two_dimensional_locality_transpose_study_stability.csv)
- [transpose gpu ms](./results/charts/transpose_gpu_ms.png)
- [transpose speedup vs naive](./results/charts/transpose_speedup_vs_naive.png)
- [transpose stability](./results/charts/transpose_stability.png)

## Interpretation
- On this GPU, the expected ordering is clear: row-major copy is the best-case access pattern, naive transpose is the worst, and both tiled transpose variants recover a large fraction of the lost locality.
- The padded tiled variant does not change the result materially in the current configuration; the main gain comes from tiling itself rather than from the extra padding adjustment.

## Limitations
- Results come from one GPU and driver stack.
- The current sweep covers one configured matrix size only, so it does not show whether the padded-vs-unpadded tiled gap changes across smaller or larger matrices.
- This experiment compares one copy path and three transpose-style kernels only; it does not include texture hardware or other layout transforms.
