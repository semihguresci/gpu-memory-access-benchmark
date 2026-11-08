# Experiment 25 Results: Spatial Binning or Clustered Culling Capstone

## Run status and provenance

- Current export: **30/30 correctness checks passed**, 6 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `17db0b4461faef1302704b841f1fef982a8e746efbaa3b069dde3bd656cc3847`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "25_spatial_binning_clustered_culling_capstone", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\25_spatial_binning_clustered_culling_capstone\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:41:17.616671Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `6` |
| row_count | `30` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| active_bin_count | 20, 64, 8 |
| bin_list_span_bytes | 32505856 |
| counter_span_bytes | 256 |
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| input_span_bytes | 1015808 |
| mean_active_bin_load | 15872.000000, 1984.000000, 6348.800000 |
| scratch_size_bytes | 33554432 |
| seed | 1476052534, 285682394, 641211877 |
| validation_mode | per_bin_sorted_entity_ids |

## Key measurements

All cases share the reported **problem_size=126976**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| coherent_append_clustered | 1 | 126976 | 5 | 0.303680 | 418124341.412 | 6.689989 |
| coherent_append_uniform_dense | 1 | 126976 | 5 | 0.274144 | 463172639.197 | 7.410762 |
| coherent_append_uniform_sparse | 1 | 126976 | 5 | 0.322112 | 394198291.278 | 6.307173 |
| global_append_clustered | 1 | 126976 | 5 | 2.448000 | 51869281.046 | 0.829908 |
| global_append_uniform_dense | 1 | 126976 | 5 | 2.563680 | 49528802.347 | 0.792461 |
| global_append_uniform_sparse | 1 | 126976 | 5 | 3.011840 | 42158946.026 | 0.674543 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114117Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114117Z_remaining_review_fixes.json).
- [spatial_binning_clustered_culling_capstone_estimated_gbps.png](./results/charts/spatial_binning_clustered_culling_capstone_estimated_gbps.png)
- [spatial_binning_clustered_culling_capstone_median_gpu_ms.png](./results/charts/spatial_binning_clustered_culling_capstone_median_gpu_ms.png)
- [spatial_binning_clustered_culling_capstone_speedup_vs_global.png](./results/charts/spatial_binning_clustered_culling_capstone_speedup_vs_global.png)
- [spatial_binning_clustered_culling_capstone_stability_ratio.png](./results/charts/spatial_binning_clustered_culling_capstone_stability_ratio.png)
- [spatial_binning_clustered_culling_capstone_relative.csv](./results/tables/spatial_binning_clustered_culling_capstone_relative.csv)
- [spatial_binning_clustered_culling_capstone_stability.csv](./results/tables/spatial_binning_clustered_culling_capstone_stability.csv)
- [spatial_binning_clustered_culling_capstone_summary.csv](./results/tables/spatial_binning_clustered_culling_capstone_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- CPU sorting is outside the measured kernel; results do not establish end-to-end sorting amortization.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
