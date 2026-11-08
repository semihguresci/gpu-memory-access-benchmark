# Experiment 02 Results: Local Size Sweep

## Run status and provenance

- Current export: **540/540 correctness checks passed**, 12 variants, 9 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `9e79327b90e0fbe960d3b18aea0b32bd82f3d7b7ddff38110aa30f02f0e9fe61`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "02_local_size_sweep", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\02_local_size_sweep\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:32:11.198960Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `108` |
| row_count | `540` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| readback_scope | copy_and_host_visibility |

## Key measurements

All cases share the reported **problem_size=2796202**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| contiguous_write_ls1024 | 1 | 2796202 | 5 | 0.032064 | 87206898702.595 | 348.827595 |
| contiguous_write_ls128 | 1 | 2796202 | 5 | 0.031424 | 88983006619.145 | 355.932026 |
| contiguous_write_ls256 | 1 | 2796202 | 5 | 0.032320 | 86516150990.099 | 346.064604 |
| contiguous_write_ls32 | 1 | 2796202 | 5 | 0.091200 | 30660109649.123 | 122.640439 |
| contiguous_write_ls512 | 1 | 2796202 | 5 | 0.032160 | 86946579601.990 | 347.786318 |
| contiguous_write_ls64 | 1 | 2796202 | 5 | 0.047776 | 58527335900.871 | 234.109344 |
| noop_ls1024 | 1 | 2796202 | 5 | 0.016128 | 0.000 | 0.000000 |
| noop_ls128 | 1 | 2796202 | 5 | 0.025248 | 0.000 | 0.000000 |
| noop_ls256 | 1 | 2796202 | 5 | 0.015648 | 0.000 | 0.000000 |
| noop_ls32 | 1 | 2796202 | 5 | 0.090272 | 0.000 | 0.000000 |
| noop_ls512 | 1 | 2796202 | 5 | 0.015136 | 0.000 | 0.000000 |
| noop_ls64 | 1 | 2796202 | 5 | 0.046784 | 0.000 | 0.000000 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113211Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113211Z_remaining_review_fixes.json).
- [benchmark_summary.png](./results/charts/benchmark_summary.png)
- [local_size_sweep_gpu_ms_vs_local_size.png](./results/charts/local_size_sweep_gpu_ms_vs_local_size.png)
- [local_size_sweep_operation_ratio_summary.png](./results/charts/local_size_sweep_operation_ratio_summary.png)
- [local_size_sweep_speedup_vs_ls64.png](./results/charts/local_size_sweep_speedup_vs_ls64.png)
- [local_size_sweep_status_overview.png](./results/charts/local_size_sweep_status_overview.png)
- [local_size_sweep_test_setup.png](./results/charts/local_size_sweep_test_setup.png)
- [local_size_sweep_throughput_vs_local_size.png](./results/charts/local_size_sweep_throughput_vs_local_size.png)
- [local_size_sweep_best_local_size.csv](./results/tables/local_size_sweep_best_local_size.csv)
- [local_size_sweep_gpu_ms_pivot.csv](./results/tables/local_size_sweep_gpu_ms_pivot.csv)
- [local_size_sweep_local_size_ranking.csv](./results/tables/local_size_sweep_local_size_ranking.csv)
- [local_size_sweep_multi_run_summary.csv](./results/tables/local_size_sweep_multi_run_summary.csv)
- [local_size_sweep_operation_ratio.csv](./results/tables/local_size_sweep_operation_ratio.csv)
- [local_size_sweep_operation_ratio_summary.csv](./results/tables/local_size_sweep_operation_ratio_summary.csv)
- [local_size_sweep_runs_index.csv](./results/tables/local_size_sweep_runs_index.csv)
- [local_size_sweep_speedup_vs_ls64.csv](./results/tables/local_size_sweep_speedup_vs_ls64.csv)
- [local_size_sweep_status_overview.csv](./results/tables/local_size_sweep_status_overview.csv)
- [local_size_sweep_summary.csv](./results/tables/local_size_sweep_summary.csv)
- [local_size_sweep_test_setup.csv](./results/tables/local_size_sweep_test_setup.csv)
- [local_size_sweep_throughput_pivot.csv](./results/tables/local_size_sweep_throughput_pivot.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Workgroup size can change occupancy and scheduling together; timing alone does not identify either mechanism.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
