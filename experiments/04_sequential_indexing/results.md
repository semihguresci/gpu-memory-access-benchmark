# Experiment 04 Results: Sequential Indexing

## Run Status
- Benchmark status: latest `full_refresh_20260329` collection completed (`600/600` row correctness pass)
- Test status: no additional tests were run during this report refresh
- GPU: `NVIDIA GeForce RTX 2080 SUPER` (Vulkan `1.4.325`, driver `2480242688`)
- Config: `--iterations 5 --warmup 2 --size 64M --label full_refresh_20260329`
- Validation layers: `disabled`
- GPU timestamps: `supported`
- Raw export timestamp (UTC): `2026-03-29T14:12:30Z`
- Latest collected run: [runs/nvidia_geforce_rtx_2080_super/20260329_141230Z_full_refresh_20260329.json](./runs/nvidia_geforce_rtx_2080_super/20260329_141230Z_full_refresh_20260329.json)
- Sweep coverage: `1` variant x `15` problem sizes x `8` dispatch counts

## Key Measurements
- At the largest tested problem size (`problem_size=16777216`), `dispatch_count=1` measured `0.322528 ms` and `416.143 GB/s`.
- The peak bandwidth point in the current run was `dispatch_count=16` at `5.057568 ms` and `424.608 GB/s`.
- Even the heaviest point, `dispatch_count=1024`, still held `416.781 GB/s`, only `1.019x` below the peak.
- Across the full `dispatch_count=1..1024` sweep at that size, the measured bandwidth stayed within about `2%`, even though total GPU time scaled with the multiplied work.

## Artifact Links
- [Raw benchmark export](./results/tables/benchmark_results.json)
- [Current summary table](./results/tables/sequential_indexing_summary.csv)
- [Status overview table](./results/tables/sequential_indexing_status_overview.csv)
- [GPU time chart](./results/charts/sequential_indexing_gpu_ms_vs_size.png)
- [GB/s chart](./results/charts/sequential_indexing_gbps_vs_size.png)

## Interpretation
- Once the workload is large and fully sequential, dispatch multiplication changes total time much more than observed bandwidth. The measured bandwidth stays very flat across the sweep in this full-refresh run.
- That keeps Experiment 04 useful as a control for later experiments that deliberately perturb memory access patterns while keeping the basic dispatch structure recognizable.

## Limitations
- Results come from one GPU and driver stack.
- This benchmark exercises only the sequential read/write baseline; it does not isolate arithmetic or synchronization overheads separately.
