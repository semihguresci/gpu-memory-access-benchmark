# Experiment 38 Results: Frustum vs Clustered Culling

## Run status and provenance

- Current export: **60/60 correctness checks passed**, 4 variants, 3 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `74458dc315b249646edb1ffbfc3531bbc57f1598863ad64d059770c269c0b405`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "38_frustum_vs_clustered_culling", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\38_frustum_vs_clustered_culling\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T13:12:22.931432Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-contention-merge` |
| provenance | `{"binary_sha256": "ef0050b469cad91296ef3383eaba2c0b56e9cbd17c496285290a7089c1c7d734", "git_revision": "b0729fe9ad71ab7e5b1ae4eb727c95a00f2ace3b", "shader_bundle_file_count": 107, "shader_bundle_sha256": "a1fd83acb9cbe7b1a2680afce8cf68b02f2c90137fa2a6d0d88260b5442b5654", "source_tree_sha256": "8e852df986f950dbbe4914cc723e350ed482807b9e469cf9f7b5184e7193f371", "working_tree_dirty": true}` |
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
| gbps_basis | estimated_source_level_storage_buffer_bytes |
| gbps_mode | estimated_global_bytes |
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| logical_payload_bytes | 10485760, 12792792, 12794336, 13977600, 4797240, 4798648, 5242880, 9596300, 9596720 |
| traffic_model | source_storage_buffer_operations_not_physical_dram |

## Key measurements

All cases share the reported **problem_size=698880**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| clustered_culling_center_clustered | 1 | 698880 | 5 | 2.457344 | 284404625.482 | 5.839182 |
| clustered_culling_wide_scene | 1 | 698880 | 5 | 2.102656 | 332379618.920 | 6.261422 |
| frustum_direct_center_clustered | 1 | 698880 | 5 | 0.884640 | 790016277.808 | 15.825014 |
| frustum_direct_wide_scene | 1 | 698880 | 5 | 0.879936 | 794239581.060 | 14.563141 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_131222Z_contention_merge.json](./runs/nvidia_geforce_rtx_2080_super/20261108_131222Z_contention_merge.json).
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
