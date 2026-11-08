# Experiment 26 Results: Warp-Level Coalescing Alignment

## Run status and provenance

- Current export: **30/30 correctness checks passed**, 6 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `c20db2dfdcf451b0a1cc4f7e8949756afb933da7d92f9fd377d45c5e8ca2db49`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "26_warp_level_coalescing_alignment", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\26_warp_level_coalescing_alignment\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:41:49.169625Z` |
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
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| output_span_bytes | 16776192 |
| physical_input_span_bytes | 16776256 |
| validation_mode | transform_value_u32 |

## Key measurements

All cases share the reported **problem_size=4194048**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| offset_0b | 1 | 4194048 | 5 | 1.498976 | 2797942061.781 | 22.383536 |
| offset_16b | 1 | 4194048 | 5 | 1.502240 | 2791862818.192 | 22.334903 |
| offset_32b | 1 | 4194048 | 5 | 1.505312 | 2786165260.092 | 22.289322 |
| offset_4b | 1 | 4194048 | 5 | 1.502688 | 2791030473.392 | 22.328244 |
| offset_64b | 1 | 4194048 | 5 | 1.503104 | 2790258026.058 | 22.322064 |
| offset_8b | 1 | 4194048 | 5 | 1.502240 | 2791862818.192 | 22.334903 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114149Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114149Z_remaining_review_fixes.json).
- [warp_level_coalescing_alignment_median_gbps.png](./results/charts/warp_level_coalescing_alignment_median_gbps.png)
- [warp_level_coalescing_alignment_median_gpu_ms.png](./results/charts/warp_level_coalescing_alignment_median_gpu_ms.png)
- [warp_level_coalescing_alignment_slowdown_vs_aligned.png](./results/charts/warp_level_coalescing_alignment_slowdown_vs_aligned.png)
- [warp_level_coalescing_alignment_stability_ratio.png](./results/charts/warp_level_coalescing_alignment_stability_ratio.png)
- [warp_level_coalescing_alignment_footprint.csv](./results/tables/warp_level_coalescing_alignment_footprint.csv)
- [warp_level_coalescing_alignment_relative.csv](./results/tables/warp_level_coalescing_alignment_relative.csv)
- [warp_level_coalescing_alignment_stability.csv](./results/tables/warp_level_coalescing_alignment_stability.csv)
- [warp_level_coalescing_alignment_summary.csv](./results/tables/warp_level_coalescing_alignment_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Offset tests use the implemented access width and placement; transaction behavior needs hardware evidence.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
