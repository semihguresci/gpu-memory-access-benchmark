# Experiment 28 Results: Device-Local vs Host-Visible Heap Placement

## Run status and provenance

- Current export: **50/50 correctness checks passed**, 10 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `da4c4ab32045286993f211976a2eabc49b9b4edefa14eb924fa2cd943a77864d`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "28_device_local_vs_host_visible_heap_placement", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\28_device_local_vs_host_visible_heap_placement\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:42:24.287451Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `10` |
| row_count | `50` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| active_buffer_count | 2, 4 |
| active_payload_bytes | 11184128, 22368256 |
| allocated_memory_bytes | 33552384 |
| buffer_span_bytes | 5592064 |
| device_destination_heap_index | 0 |
| device_destination_memory_property_flags | 1 |
| device_destination_memory_type_index | 1 |
| device_source_heap_index | 0 |
| device_source_memory_property_flags | 1 |
| device_source_memory_type_index | 1 |
| gbps_basis | useful_kernel_payload_over_gpu_path_time |
| gpu_time_scope | dispatch_chain_timestamps, sum_of_upload_dispatch_chain_readback_timestamps, sum_of_upload_kernel_readback_timestamps |
| host_destination_heap_index | 1 |
| host_destination_memory_property_flags | 6 |
| host_destination_memory_type_index | 3 |
| host_source_heap_index | 1 |
| host_source_memory_property_flags | 6 |
| host_source_memory_type_index | 3 |
| live_buffer_count | 6 |
| live_payload_bytes | 33552384 |
| payload_bytes_per_element | 8 |
| physical_residency_measured | false |
| scratch_budget_scope | buffer_payload_excludes_allocation_padding |
| scratch_size_bytes | 33554432 |
| staging_destination_heap_index | 1 |
| staging_destination_memory_property_flags | 6 |
| staging_destination_memory_type_index | 3 |
| staging_source_heap_index | 1 |
| staging_source_memory_property_flags | 6 |
| staging_source_memory_type_index | 3 |

## Key measurements

All cases share the reported **problem_size=1398016**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| device_local_staged | 1 | 1398016 | 5 | 0.889984 | 1570832734.072 | 12.566662 |
| device_local_staged_reuse_k16 | 16 | 1398016 | 5 | 1.325184 | 16879358640.008 | 135.034869 |
| device_local_staged_reuse_k2 | 2 | 1398016 | 5 | 0.919104 | 3042127985.516 | 24.337024 |
| device_local_staged_reuse_k4 | 4 | 1398016 | 5 | 0.976480 | 5726757332.459 | 45.814059 |
| device_local_staged_reuse_k64 | 64 | 1398016 | 5 | 2.726688 | 32813810747.691 | 262.510486 |
| host_visible_direct | 1 | 1398016 | 5 | 0.507968 | 2752173365.251 | 22.017387 |
| host_visible_direct_reuse_k16 | 16 | 1398016 | 5 | 8.053472 | 2777467407.846 | 22.219739 |
| host_visible_direct_reuse_k2 | 2 | 1398016 | 5 | 1.012288 | 2762091420.623 | 22.096731 |
| host_visible_direct_reuse_k4 | 4 | 1398016 | 5 | 2.017568 | 2771685514.441 | 22.173484 |
| host_visible_direct_reuse_k64 | 64 | 1398016 | 5 | 33.038464 | 2708147206.844 | 21.665178 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114224Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114224Z_remaining_review_fixes.json).
- [heap_placement_gpu_path_gbps.png](./results/charts/heap_placement_gpu_path_gbps.png)
- [heap_placement_gpu_path_ms.png](./results/charts/heap_placement_gpu_path_ms.png)
- [heap_placement_harness_wall_clock_ms.png](./results/charts/heap_placement_harness_wall_clock_ms.png)
- [heap_placement_matched_kernel_count_ratio.png](./results/charts/heap_placement_matched_kernel_count_ratio.png)
- [heap_placement_reuse_amortization.png](./results/charts/heap_placement_reuse_amortization.png)
- [device_local_vs_host_visible_heap_placement_relative.csv](./results/tables/device_local_vs_host_visible_heap_placement_relative.csv)
- [device_local_vs_host_visible_heap_placement_stability.csv](./results/tables/device_local_vs_host_visible_heap_placement_stability.csv)
- [device_local_vs_host_visible_heap_placement_summary.csv](./results/tables/device_local_vs_host_visible_heap_placement_summary.csv)
- [device_local_vs_host_visible_reuse_amortization.csv](./results/tables/device_local_vs_host_visible_reuse_amortization.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Six buffers remain live within the payload budget; active direct/staged paths use two/four. Allocation padding is reported separately. Actual property flags may identify the same heap for both policies. GPU time sums stage intervals and excludes CPU submission gaps; GB/s uses useful kernel payload.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
