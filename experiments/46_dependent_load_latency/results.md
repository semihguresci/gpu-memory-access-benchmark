# Experiment 46 Results: Dependent Load Latency

## Run status and provenance

- Current export: **100/100 correctness checks passed**, 5 variants, 4 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `299aac1c5d66455206976a2c69fbe52146a53b5d0446708410fc7f3c553ae410`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "46_dependent_load_latency", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\46_dependent_load_latency\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:46:36.865303Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `20` |
| row_count | `100` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| active_payload_bytes | 1048576, 16777216, 33554432, 4194304 |
| buffer_span_bytes | 1048576, 262144, 4194304, 8388608 |
| chain_count | 1, 16, 2, 4, 8 |
| chain_count_source | pipeline_specialization |
| chain_issue_order | interleaved_steps |
| chain_start_stride | 2654435761 |
| chain_state_storage | explicit_scalars |
| chain_steps | 16, 32, 4, 64, 8 |
| estimated_request_bytes | 17039360, 272629760, 545259520, 68157440 |
| gbps_mode | estimated_shader_request_bytes |
| live_allocation_bytes | 33554432 |
| live_buffer_count | 4 |
| live_payload_bytes | 33554432 |
| permutation_seed | 2495676398 |
| resident_buffer_span_bytes | 8388608 |
| scratch_size_bytes | 33554432 |

## Key measurements

All cases share the reported **problem_size=2097152**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| chain_1 | 1 | 2097152 | 5 | 24.754816 | 84716929.425 | 22.026402 |
| chain_16 | 1 | 2097152 | 5 | 19.445856 | 107845702.447 | 28.039883 |
| chain_2 | 1 | 2097152 | 5 | 24.429440 | 85845275.209 | 22.319772 |
| chain_4 | 1 | 2097152 | 5 | 23.818464 | 88047323.287 | 22.892304 |
| chain_8 | 1 | 2097152 | 5 | 22.510496 | 93163295.913 | 24.222457 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114636Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114636Z_remaining_review_fixes.json).
- [dependent_load_latency_ms_vs_size.png](./results/charts/dependent_load_latency_ms_vs_size.png)
- [dependent_load_latency_ns_per_load_vs_size.png](./results/charts/dependent_load_latency_ns_per_load_vs_size.png)
- [dependent_load_latency_relative.png](./results/charts/dependent_load_latency_relative.png)
- [dependent_load_latency_stability.png](./results/charts/dependent_load_latency_stability.png)
- [dependent_load_latency_checksums.csv](./results/tables/dependent_load_latency_checksums.csv)
- [dependent_load_latency_per_load_latency.csv](./results/tables/dependent_load_latency_per_load_latency.csv)
- [dependent_load_latency_relative.csv](./results/tables/dependent_load_latency_relative.csv)
- [dependent_load_latency_stability.csv](./results/tables/dependent_load_latency_stability.csv)
- [dependent_load_latency_summary.csv](./results/tables/dependent_load_latency_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Specialized kernels interleave 1/2/4/8/16 independent pointer chains at 64 requested loads per invocation. Aggregate ns/load includes scheduling, occupancy, and cache effects; it is not isolated hardware latency.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
