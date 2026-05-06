# Experiment 34 Results: Radix Sort on GPU

## Run Status
- Benchmark status: latest `20260428_170950Z` collection completed (`60/60` row correctness pass across `12` benchmark cases)
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-04-28T17:09:50Z`
- Latest collected run: [runs archive](./runs/nvidia_geforce_rtx_2080_super/20260428_170950Z.json)
- Sweep coverage: `2` variants x `6` problem sizes (`262144` through `8388608` elements)

## Key Measurements
- All `12` benchmark cases are represented in the refreshed export.
- At the largest tested `element_count=8388608`, `4bit_8pass` reached `166.200864 ms` median GPU time versus `452.984416 ms` for `8bit_4pass`, a `2.725524x` speedup. Median estimated GB/s: `8.378473` versus `2.370382`.
- The largest p95-to-median GPU-time ratio in the current stability table is `1.074403` (`4bit_8pass`, `element_count=524288`).

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [radix sort gpu relative](./results/tables/radix_sort_gpu_relative.csv)
- [radix sort gpu stability](./results/tables/radix_sort_gpu_stability.csv)
- [radix sort gpu summary](./results/tables/radix_sort_gpu_summary.csv)
- [radix sort gpu median ms](./results/charts/radix_sort_gpu_ms.png)
- [radix sort throughput](./results/charts/radix_sort_throughput.png)
- [radix sort stability](./results/charts/radix_sort_stability.png)

## Interpretation
- In this refreshed sweep, `4bit_8pass` is consistently faster than `8bit_4pass` at every tested size on the RTX 2080 SUPER, with the largest-size case still showing a substantial gap.
- The current five-sample runs are also fairly tight, so the ranking in this dataset does not look like a one-off outlier.

## Limitations
- Results come from one GPU and driver stack.
- Reported GB/s values are based on the experiment's estimated global-byte accounting (`estimated_global_bytes_per_sort` in run notes), so comparisons are safest within this experiment.
- The sweep covers only the current `4bit_8pass` and `8bit_4pass` implementations; other radix widths, payload formats, or devices may shift the tradeoff.
