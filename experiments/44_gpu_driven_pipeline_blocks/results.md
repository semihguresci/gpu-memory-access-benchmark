# Experiment 44 Results: GPU-Driven Pipeline Blocks

## Run status and provenance

- Current export: **20/20 correctness checks passed**, 2 variants, 2 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `a42c9bdc146225dcfdbb013546c5223ae5a4a364c4ed78cc55b4891aad4edccd`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "44_gpu_driven_pipeline_blocks", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\44_gpu_driven_pipeline_blocks\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:46:12.023746Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `4` |
| row_count | `20` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| estimated_traffic_formula | 16N, 32N_plus_4V |
| gbps_mode | logical_payload_bytes |
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| logical_payload_bytes | 16777216, 8388608 |
| traffic_model | source_storage_buffer_operations_not_physical_dram |
| visible_count | 1363386, 681389 |

## Key measurements

All cases share the reported **problem_size=2097152**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| fused_single_dispatch | 1 | 2097152 | 5 | 42.437280 | 49417681.812 | 0.395341 |
| staged_three_dispatch | 3 | 2097152 | 5 | 29.346208 | 71462452.662 | 0.571700 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114612Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114612Z_remaining_review_fixes.json).
- [gpu_driven_pipeline_blocks_ms.png](./results/charts/gpu_driven_pipeline_blocks_ms.png)
- [gpu_driven_pipeline_blocks_relative.png](./results/charts/gpu_driven_pipeline_blocks_relative.png)
- [gpu_driven_pipeline_blocks_stability.png](./results/charts/gpu_driven_pipeline_blocks_stability.png)
- [gpu_driven_pipeline_blocks_throughput.png](./results/charts/gpu_driven_pipeline_blocks_throughput.png)
- [gpu_driven_pipeline_blocks_relative.csv](./results/tables/gpu_driven_pipeline_blocks_relative.csv)
- [gpu_driven_pipeline_blocks_stability.csv](./results/tables/gpu_driven_pipeline_blocks_stability.csv)
- [gpu_driven_pipeline_blocks_summary.csv](./results/tables/gpu_driven_pipeline_blocks_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- The staged/fused cull/bucket/emit-style workload is a proxy without full compaction or indirect execution. Useful payload and modeled global traffic are distinct metrics.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
