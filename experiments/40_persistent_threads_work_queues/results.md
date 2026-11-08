# Experiment 40 Results: Persistent Threads and Work Queues

## Run status and provenance

- Current export: **60/60 correctness checks passed**, 4 variants, 3 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `260d07164c49dfffcc67c075c43f483edf3178a7b7d6654b5e01022e29e6c5c4`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "40_persistent_threads_work_queues", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\40_persistent_threads_work_queues\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T13:12:28.487064Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-contention-merge` |
| provenance | `{"binary_sha256": "ef0050b469cad91296ef3383eaba2c0b56e9cbd17c496285290a7089c1c7d734", "git_revision": "b0729fe9ad71ab7e5b1ae4eb727c95a00f2ace3b", "shader_bundle_file_count": 107, "shader_bundle_sha256": "a1fd83acb9cbe7b1a2680afce8cf68b02f2c90137fa2a6d0d88260b5442b5654", "source_tree_sha256": "8e852df986f950dbbe4914cc723e350ed482807b9e469cf9f7b5184e7193f371", "working_tree_dirty": true}` |
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
| gbps_basis | estimated_source_level_storage_buffer_bytes |
| gbps_mode | estimated_global_bytes |
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| logical_payload_bytes | 2097152, 22368256, 8388608 |
| traffic_model | source_storage_buffer_operations_not_physical_dram |

## Key measurements

All cases share the reported **problem_size=2796032**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| persistent_queue_skewed_tail | 1 | 2796032 | 5 | 1.119072 | 2498527351.234 | 20.067213 |
| persistent_queue_uniform_cost | 1 | 2796032 | 5 | 1.266720 | 2207300745.232 | 17.728192 |
| static_partitioned_skewed_tail | 1 | 2796032 | 5 | 1.001248 | 2792546901.467 | 22.340375 |
| static_partitioned_uniform_cost | 1 | 2796032 | 5 | 1.000768 | 2793886295.325 | 22.351090 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_131228Z_contention_merge.json](./runs/nvidia_geforce_rtx_2080_super/20261108_131228Z_contention_merge.json).
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
- Persistent scheduling claims one workgroup-sized chunk at a time and includes a workgroup barrier before reusing the shared claim. GB/s estimates source-level storage-buffer traffic with 8 bytes per global atomic read-modify-write; it does not measure physical DRAM traffic. Results depend on task distribution and do not cover all queue granularities.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
