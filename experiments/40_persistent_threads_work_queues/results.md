# Experiment 40 Results: Persistent Threads and Work Queues

## Run status and provenance

- Current export: **60/60 correctness checks passed**, 4 variants, 3 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `fa7220e7054ce8d8f8c94f9d969ff4dc7f1c7ab8f48a8982c34003b325da21e8`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "40_persistent_threads_work_queues", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\40_persistent_threads_work_queues\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:45:43.510764Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `12` |
| row_count | `60` |
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
| logical_payload_bytes | 2097152, 22368256, 8388608 |

## Key measurements

All cases share the reported **problem_size=2796032**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| persistent_queue_skewed_tail | 1 | 2796032 | 5 | 4.231648 | 660743048.571 | 7.959891 |
| persistent_queue_uniform_cost | 1 | 2796032 | 5 | 6.238080 | 448219965.117 | 5.399651 |
| static_partitioned_skewed_tail | 1 | 2796032 | 5 | 1.001088 | 2792993223.373 | 22.343946 |
| static_partitioned_uniform_cost | 1 | 2796032 | 5 | 1.000832 | 2793707635.247 | 22.349661 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114543Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114543Z_remaining_review_fixes.json).
- [persistent_threads_work_queues_ms.png](./results/charts/persistent_threads_work_queues_ms.png)
- [persistent_threads_work_queues_relative.png](./results/charts/persistent_threads_work_queues_relative.png)
- [persistent_threads_work_queues_stability.png](./results/charts/persistent_threads_work_queues_stability.png)
- [persistent_threads_work_queues_throughput.png](./results/charts/persistent_threads_work_queues_throughput.png)
- [persistent_threads_work_queues_relative.csv](./results/tables/persistent_threads_work_queues_relative.csv)
- [persistent_threads_work_queues_stability.csv](./results/tables/persistent_threads_work_queues_stability.csv)
- [persistent_threads_work_queues_summary.csv](./results/tables/persistent_threads_work_queues_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Persistent scheduling includes per-lane task-claim costs; results depend on task distribution and do not cover all queue granularities.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
