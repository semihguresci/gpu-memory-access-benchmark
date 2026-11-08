# Experiment 16 Results: Shared Memory Tiling

## Run status and provenance

- Current export: **40/40 correctness checks passed**, 8 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `b554f6c69e71c0467708c0994e8dbe36168385216c0f6a3eef831c2a33ffcdd4`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "16_shared_memory_tiling", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\16_shared_memory_tiling\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:37:46.948772Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `8` |
| row_count | `40` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| active_tile_span_elements | 258, 264, 272, 288 |
| gpu_time_scope | dispatch_and_host_visibility |
| scratch_size_bytes | 33554432 |
| validation_mode | exact_uint32 |

## Key measurements

All cases share the reported **problem_size=4194048**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| direct_global_r1 | 1 | 4194048 | 5 | 1.508256 | 2780726879.257 | 44.491630 |
| direct_global_r16 | 1 | 4194048 | 5 | 1.506592 | 2783798135.129 | 378.596546 |
| direct_global_r4 | 1 | 4194048 | 5 | 1.505920 | 2785040373.991 | 111.401615 |
| direct_global_r8 | 1 | 4194048 | 5 | 1.506848 | 2783325192.720 | 200.399414 |
| shared_tiled_r1 | 1 | 4194048 | 5 | 1.557408 | 2692966775.566 | 21.627889 |
| shared_tiled_r16 | 1 | 4194048 | 5 | 1.500608 | 2794899134.218 | 23.756643 |
| shared_tiled_r4 | 1 | 4194048 | 5 | 1.557920 | 2692081750.026 | 21.873164 |
| shared_tiled_r8 | 1 | 4194048 | 5 | 1.557088 | 2693520212.088 | 22.221542 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113746Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113746Z_remaining_review_fixes.json).
- [shared_memory_tiling_estimated_gbps.png](./results/charts/shared_memory_tiling_estimated_gbps.png)
- [shared_memory_tiling_median_gpu_ms.png](./results/charts/shared_memory_tiling_median_gpu_ms.png)
- [shared_memory_tiling_speedup_vs_direct.png](./results/charts/shared_memory_tiling_speedup_vs_direct.png)
- [shared_memory_tiling_stability_ratio.png](./results/charts/shared_memory_tiling_stability_ratio.png)
- [shared_memory_tiling_relative.csv](./results/tables/shared_memory_tiling_relative.csv)
- [shared_memory_tiling_stability.csv](./results/tables/shared_memory_tiling_stability.csv)
- [shared_memory_tiling_summary.csv](./results/tables/shared_memory_tiling_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Stencil radius changes reuse and useful work; compare matched-radius direct and tiled cases.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
