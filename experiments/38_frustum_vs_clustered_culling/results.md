# Experiment 38 Results: Frustum vs Clustered Culling

## Run status and provenance

- Current export: **60/60 correctness checks passed**, 4 variants, 3 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `1920714582e0e92c6e115ea95f9bde6cfa1dc246be5d886230265c834023c0e7`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "38_frustum_vs_clustered_culling", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\38_frustum_vs_clustered_culling\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:45:27.661068Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `12` |
| row_count | `60` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| expected_visible_count | 150734, 151086, 262144, 301923, 302028, 402678, 403064, 524288, 698880 |
| gbps_mode | estimated_global_bytes |
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| logical_payload_bytes | 10485760, 12792792, 12794336, 13977600, 4797240, 4798648, 5242880, 9596300, 9596720 |

## Key measurements

All cases share the reported **problem_size=698880**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| clustered_culling_center_clustered | 1 | 698880 | 5 | 11.666784 | 59903397.543 | 1.677295 |
| clustered_culling_wide_scene | 1 | 698880 | 5 | 15.554400 | 44931337.757 | 1.029860 |
| frustum_direct_center_clustered | 1 | 698880 | 5 | 2.959776 | 236125977.101 | 5.667023 |
| frustum_direct_wide_scene | 1 | 698880 | 5 | 1.579904 | 442355991.250 | 9.116696 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114527Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114527Z_remaining_review_fixes.json).
- [frustum_vs_clustered_culling_ms.png](./results/charts/frustum_vs_clustered_culling_ms.png)
- [frustum_vs_clustered_culling_relative.png](./results/charts/frustum_vs_clustered_culling_relative.png)
- [frustum_vs_clustered_culling_stability.png](./results/charts/frustum_vs_clustered_culling_stability.png)
- [frustum_vs_clustered_culling_throughput.png](./results/charts/frustum_vs_clustered_culling_throughput.png)
- [frustum_vs_clustered_culling_relative.csv](./results/tables/frustum_vs_clustered_culling_relative.csv)
- [frustum_vs_clustered_culling_stability.csv](./results/tables/frustum_vs_clustered_culling_stability.csv)
- [frustum_vs_clustered_culling_summary.csv](./results/tables/frustum_vs_clustered_culling_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- This is a visibility append/bin-count proxy; it does not establish downstream culling or rendering savings.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
