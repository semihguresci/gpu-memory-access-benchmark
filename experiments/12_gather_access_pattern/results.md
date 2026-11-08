# Experiment 12 Results: Gather Access Pattern

## Run status and provenance

- Current export: **20/20 correctness checks passed**, 4 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `2f0bd38435699ee50213c507ddc8cfcd5df8a6767885ca1087d331e6ceb2fb93`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "12_gather_access_pattern", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\12_gather_access_pattern\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:35:54.360382Z` |
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
| allocated_span_bytes | 11184808 |
| gpu_time_scope | dispatch_and_host_visibility |
| per_buffer_budget_bytes | 11184810 |
| physical_span_bytes | 11184808 |
| resident_buffer_payload_bytes | 33554424 |
| scratch_size_bytes | 33554432 |
| seed | 1821285621 |
| validation_mode | exact_uint32 |

## Key measurements

All cases share the reported **problem_size=2796202**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| block_coherent_32 | 1 | 2796202 | 5 | 1.740128 | 1606894435.352 | 19.282733 |
| clustered_random_256 | 1 | 2796202 | 5 | 1.749216 | 1598545862.832 | 19.182550 |
| identity | 1 | 2796202 | 5 | 1.731040 | 1615330668.269 | 19.383968 |
| random_permutation | 1 | 2796202 | 5 | 10.147232 | 275563030.391 | 3.306756 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113554Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113554Z_remaining_review_fixes.json).
- [gather_access_pattern_median_gbps.png](./results/charts/gather_access_pattern_median_gbps.png)
- [gather_access_pattern_median_gpu_ms.png](./results/charts/gather_access_pattern_median_gpu_ms.png)
- [gather_access_pattern_slowdown_vs_identity.png](./results/charts/gather_access_pattern_slowdown_vs_identity.png)
- [gather_access_pattern_stability_ratio.png](./results/charts/gather_access_pattern_stability_ratio.png)
- [gather_access_pattern_relative.csv](./results/tables/gather_access_pattern_relative.csv)
- [gather_access_pattern_stability.csv](./results/tables/gather_access_pattern_stability.csv)
- [gather_access_pattern_summary.csv](./results/tables/gather_access_pattern_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Gather patterns may differ in duplicate rate and unique working set; the row notes retain total scratch and spans.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
