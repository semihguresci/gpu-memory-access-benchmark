# Experiment 01 Results: Dispatch Basics

## Run status and provenance

- Current export: **1040/1040 correctness checks passed**, 2 variants, 13 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `30cda141d8377acc6b728642c03f94586f44da59ce3201c76a329b7c1040c7a7`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "01_dispatch_basics", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\01_dispatch_basics\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:31:33.189396Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `208` |
| row_count | `1040` |
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
| contiguous_write | 1 | 2796202 | 5 | 0.031872 | 87732241465.863 | 350.928966 |
| contiguous_write | 4 | 2796202 | 5 | 0.185248 | 60377483157.713 | 241.509933 |
| contiguous_write | 16 | 2796202 | 5 | 0.734592 | 60903511064.645 | 243.614044 |
| contiguous_write | 64 | 2796202 | 5 | 2.968160 | 60292210662.498 | 241.168843 |
| contiguous_write | 128 | 2796202 | 5 | 5.945056 | 60203613893.629 | 240.814456 |
| contiguous_write | 256 | 2796202 | 5 | 11.903168 | 60137579508.245 | 240.550318 |
| contiguous_write | 512 | 2796202 | 5 | 18.258592 | 78409957569.565 | 313.639830 |
| contiguous_write | 1024 | 2796202 | 5 | 28.619648 | 100047032304.520 | 400.188129 |
| noop | 1 | 2796202 | 5 | 0.046816 | 0.000 | 0.000000 |
| noop | 4 | 2796202 | 5 | 0.178176 | 0.000 | 0.000000 |
| noop | 16 | 2796202 | 5 | 0.703488 | 0.000 | 0.000000 |
| noop | 64 | 2796202 | 5 | 2.806400 | 0.000 | 0.000000 |
| noop | 128 | 2796202 | 5 | 5.610432 | 0.000 | 0.000000 |
| noop | 256 | 2796202 | 5 | 11.215040 | 0.000 | 0.000000 |
| noop | 512 | 2796202 | 5 | 20.003232 | 0.000 | 0.000000 |
| noop | 1024 | 2796202 | 5 | 32.844512 | 0.000 | 0.000000 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113133Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113133Z_remaining_review_fixes.json).
- [benchmark_summary.png](./results/charts/benchmark_summary.png)
- [dispatch_basics_cross_device_comparison.png](./results/charts/dispatch_basics_cross_device_comparison.png)
- [dispatch_basics_nvidia_geforce_rtx_2080_super_2559967232_1_4_341_time_throughput.png](./results/charts/dispatch_basics_nvidia_geforce_rtx_2080_super_2559967232_1_4_341_time_throughput.png)
- [dispatch_basics_operation_ratio_by_device.png](./results/charts/dispatch_basics_operation_ratio_by_device.png)
- [dispatch_basics_peak_throughput.png](./results/charts/dispatch_basics_peak_throughput.png)
- [dispatch_basics_ratio_summary.png](./results/charts/dispatch_basics_ratio_summary.png)
- [dispatch_basics_representative_points.png](./results/charts/dispatch_basics_representative_points.png)
- [dispatch_basics_status_overview.png](./results/charts/dispatch_basics_status_overview.png)
- [dispatch_basics_test_setup.png](./results/charts/dispatch_basics_test_setup.png)
- [dispatch_basics_time_throughput.png](./results/charts/dispatch_basics_time_throughput.png)
- [dispatch_basics_best_dispatch.csv](./results/tables/dispatch_basics_best_dispatch.csv)
- [dispatch_basics_best_dispatch_by_device.csv](./results/tables/dispatch_basics_best_dispatch_by_device.csv)
- [dispatch_basics_device_chart_index.csv](./results/tables/dispatch_basics_device_chart_index.csv)
- [dispatch_basics_device_summary.csv](./results/tables/dispatch_basics_device_summary.csv)
- [dispatch_basics_gpu_ms_pivot.csv](./results/tables/dispatch_basics_gpu_ms_pivot.csv)
- [dispatch_basics_multi_run_summary.csv](./results/tables/dispatch_basics_multi_run_summary.csv)
- [dispatch_basics_operation_ratio_by_device.csv](./results/tables/dispatch_basics_operation_ratio_by_device.csv)
- [dispatch_basics_peak_throughput.csv](./results/tables/dispatch_basics_peak_throughput.csv)
- [dispatch_basics_ratio_summary.csv](./results/tables/dispatch_basics_ratio_summary.csv)
- [dispatch_basics_representative_points.csv](./results/tables/dispatch_basics_representative_points.csv)
- [dispatch_basics_runs_index.csv](./results/tables/dispatch_basics_runs_index.csv)
- [dispatch_basics_status_overview.csv](./results/tables/dispatch_basics_status_overview.csv)
- [dispatch_basics_summary.csv](./results/tables/dispatch_basics_summary.csv)
- [dispatch_basics_test_setup.csv](./results/tables/dispatch_basics_test_setup.csv)
- [dispatch_basics_throughput_pivot.csv](./results/tables/dispatch_basics_throughput_pivot.csv)
- Omitted 1 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Repeated dispatches include compute dependencies; timing includes their synchronization cost.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
