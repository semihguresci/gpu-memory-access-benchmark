# Experiment 39 Results: Tiled Light Assignment

## Run status and provenance

- Current export: **90/90 correctness checks passed**, 6 variants, 3 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `60f11fad2b546f4988e6729496efc1f4cc6fe55af44c85d791459c60188519ca`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "39_tiled_light_assignment", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\39_tiled_light_assignment\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:45:37.799430Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `18` |
| row_count | `90` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| gbps_mode | estimated_global_bytes |
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| light_heap | 1 |
| light_memory_flags | 6 |
| light_memory_type | 3 |
| logical_payload_bytes | 11183108, 4195332, 8389636 |
| tile_count_heap | 1 |
| tile_count_memory_flags | 6 |
| tile_count_memory_type | 3 |

## Key measurements

All cases share the reported **problem_size=698880**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| light_atomic_center_clustered | 1 | 698880 | 5 | 80.394560 | 8693125.505 | 0.948643 |
| light_atomic_uniform_lights | 1 | 698880 | 5 | 64.368000 | 10857568.978 | 0.889676 |
| tile_parallel_shared_center_clustered | 1 | 698880 | 5 | 1.791872 | 390027859.133 | 1597.554685 |
| tile_parallel_shared_uniform_lights | 1 | 698880 | 5 | 1.789248 | 390599849.769 | 1599.897559 |
| tile_serial_center_clustered | 1 | 698880 | 5 | 151.774848 | 4604715.532 | 18.860922 |
| tile_serial_uniform_lights | 1 | 698880 | 5 | 151.790080 | 4604253.453 | 18.859029 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114537Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114537Z_remaining_review_fixes.json).
- [tiled_light_assignment_ms.png](./results/charts/tiled_light_assignment_ms.png)
- [tiled_light_assignment_relative.png](./results/charts/tiled_light_assignment_relative.png)
- [tiled_light_assignment_stability.png](./results/charts/tiled_light_assignment_stability.png)
- [tiled_light_assignment_throughput.png](./results/charts/tiled_light_assignment_throughput.png)
- [tiled_light_assignment_relative.csv](./results/tables/tiled_light_assignment_relative.csv)
- [tiled_light_assignment_stability.csv](./results/tables/tiled_light_assignment_stability.csv)
- [tiled_light_assignment_summary.csv](./results/tables/tiled_light_assignment_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Strategies examine different candidate sets. GB/s is estimated source-level traffic, including strategy-specific auxiliary accesses and modeled atomic operations, rather than observed DRAM bytes; overlap totals are validated outside timing.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
