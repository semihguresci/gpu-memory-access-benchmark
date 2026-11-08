# Experiment 08 Results: std430 vs std140 vs Packed

## Run status and provenance

- Current export: **75/75 correctness checks passed**, 3 variants, 5 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `dc8b6295a96ff3f78f5c0889e0e5144ec006706a7edf7059be8d97c6aa9e158d`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "08_std430_std140_packed", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\08_std430_std140_packed\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:34:46.570485Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `15` |
| row_count | `75` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| gpu_time_scope | dispatch_and_host_visibility |

## Key measurements

All cases share the reported **problem_size=131072**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| packed | 1 | 131072 | 5 | 7.212800 | 18172138.421 | 2.326034 |
| std140 | 1 | 131072 | 5 | 7.222464 | 18147823.236 | 2.322921 |
| std430 | 1 | 131072 | 5 | 6.662304 | 19673674.453 | 2.518230 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113446Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113446Z_remaining_review_fixes.json).
- [benchmark_summary.png](./results/charts/benchmark_summary.png)
- [std430_std140_packed_bandwidth_efficiency_vs_size.png](./results/charts/std430_std140_packed_bandwidth_efficiency_vs_size.png)
- [std430_std140_packed_gbps_vs_size.png](./results/charts/std430_std140_packed_gbps_vs_size.png)
- [std430_std140_packed_gpu_ms_vs_size.png](./results/charts/std430_std140_packed_gpu_ms_vs_size.png)
- [std430_std140_packed_layout_footprint.png](./results/charts/std430_std140_packed_layout_footprint.png)
- [std430_std140_packed_logical_gbps_vs_size.png](./results/charts/std430_std140_packed_logical_gbps_vs_size.png)
- [std430_std140_packed_layout_overview.csv](./results/tables/std430_std140_packed_layout_overview.csv)
- [std430_std140_packed_multi_run_summary.csv](./results/tables/std430_std140_packed_multi_run_summary.csv)
- [std430_std140_packed_relative_to_std430.csv](./results/tables/std430_std140_packed_relative_to_std430.csv)
- [std430_std140_packed_runs_index.csv](./results/tables/std430_std140_packed_runs_index.csv)
- [std430_std140_packed_status_overview.csv](./results/tables/std430_std140_packed_status_overview.csv)
- [std430_std140_packed_summary.csv](./results/tables/std430_std140_packed_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- These buffer layouts do not cover every feature-gated Vulkan scalar-layout mode.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
