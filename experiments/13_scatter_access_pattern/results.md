# Experiment 13 Results: Scatter Access Pattern

## Run status and provenance

- Current export: **15/15 correctness checks passed**, 3 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `109d4563eab832ce98619c7955d8b61ddbd2dd7afca35a79026db2f595f87c14`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "13_scatter_access_pattern", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\13_scatter_access_pattern\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:36:11.359864Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `3` |
| row_count | `15` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| active_target_count | 1048576, 4194304, 524288 |
| allocated_span_bytes | 16777216 |
| gpu_time_scope | dispatch_and_host_visibility |
| physical_span_bytes | 16777216 |
| seed | 1369949145 |
| validation_mode | exact_uint32_histogram |

## Key measurements

All cases share the reported **problem_size=4194304**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| clustered_hotset_32 | 1 | 4194304 | 5 | 53.086816 | 79008392.592 | 0.948101 |
| random_collision_x4 | 1 | 4194304 | 5 | 23.493760 | 178528426.271 | 2.142341 |
| unique_permutation | 1 | 4194304 | 5 | 23.374880 | 179436386.411 | 2.153237 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113611Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113611Z_remaining_review_fixes.json).
- [scatter_access_pattern_median_gbps.png](./results/charts/scatter_access_pattern_median_gbps.png)
- [scatter_access_pattern_median_gpu_ms.png](./results/charts/scatter_access_pattern_median_gpu_ms.png)
- [scatter_access_pattern_slowdown_vs_unique_permutation.png](./results/charts/scatter_access_pattern_slowdown_vs_unique_permutation.png)
- [scatter_access_pattern_stability_ratio.png](./results/charts/scatter_access_pattern_stability_ratio.png)
- [scatter_access_pattern_contention.csv](./results/tables/scatter_access_pattern_contention.csv)
- [scatter_access_pattern_relative.csv](./results/tables/scatter_access_pattern_relative.csv)
- [scatter_access_pattern_stability.csv](./results/tables/scatter_access_pattern_stability.csv)
- [scatter_access_pattern_summary.csv](./results/tables/scatter_access_pattern_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Atomic distribution changes contention; modeled atomic bytes do not establish physical bus traffic.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
