# Experiment 14 Results: Read Reuse and Cache Locality

## Run status and provenance

- Current export: **25/25 correctness checks passed**, 5 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `84eef77428464756e37dd2b4223e277626a18ca28f961f33fb95c2edfd82ba69`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "14_read_reuse_cache_locality", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\14_read_reuse_cache_locality\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:36:37.620119Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `5` |
| row_count | `25` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| destination_span_bytes | 13421768 |
| gpu_time_scope | dispatch_and_host_visibility |
| index_span_bytes | 13421768 |
| per_buffer_budget_bytes | 13421772 |
| scratch_size_bytes | 33554432 |
| source_span_bytes | 6710884 |
| validation_mode | exact_uint32 |

## Key measurements

All cases share the reported **problem_size=3355442**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| reuse_distance_1 | 1 | 3355442 | 5 | 1.570720 | 2136244524.804 | 25.634934 |
| reuse_distance_256 | 1 | 3355442 | 5 | 1.574944 | 2130515116.728 | 25.566181 |
| reuse_distance_32 | 1 | 3355442 | 5 | 1.575328 | 2129995785.005 | 25.559949 |
| reuse_distance_4096 | 1 | 3355442 | 5 | 1.573152 | 2132942017.046 | 25.595304 |
| reuse_distance_full_span | 1 | 3355442 | 5 | 2.080288 | 1612969934.932 | 19.355639 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113637Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113637Z_remaining_review_fixes.json).
- [read_reuse_cache_locality_median_gbps.png](./results/charts/read_reuse_cache_locality_median_gbps.png)
- [read_reuse_cache_locality_median_gpu_ms.png](./results/charts/read_reuse_cache_locality_median_gpu_ms.png)
- [read_reuse_cache_locality_speedup_vs_full_span.png](./results/charts/read_reuse_cache_locality_speedup_vs_full_span.png)
- [read_reuse_cache_locality_stability_ratio.png](./results/charts/read_reuse_cache_locality_stability_ratio.png)
- [read_reuse_cache_locality_locality.csv](./results/tables/read_reuse_cache_locality_locality.csv)
- [read_reuse_cache_locality_relative.csv](./results/tables/read_reuse_cache_locality_relative.csv)
- [read_reuse_cache_locality_stability.csv](./results/tables/read_reuse_cache_locality_stability.csv)
- [read_reuse_cache_locality_summary.csv](./results/tables/read_reuse_cache_locality_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Invocation scheduling is not a guarantee of ordered temporal reuse or cache residency.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
