# Experiment 23 Results: Histogram and Atomic Contention

## Run status and provenance

- Current export: **30/30 correctness checks passed**, 6 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `748dbdffc8af98128a402fafa46b08a02ac0948d839122481177542942f07aea`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "23_histogram_atomic_contention", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\23_histogram_atomic_contention\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:41:02.276810Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `6` |
| row_count | `30` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| histogram_span_bytes | 1024 |
| input_span_bytes | 33553408 |
| scratch_size_bytes | 33554432 |
| seed | 2505432559, 3551032828, 3835118275 |
| validation_mode | exact_histogram |

## Key measurements

All cases share the reported **problem_size=8388352**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| global_atomics_hot_bin_90 | 1 | 8388352 | 5 | 235.375424 | 35638181.155 | 0.285105 |
| global_atomics_mixed_hotset_75 | 1 | 8388352 | 5 | 121.553696 | 69009435.962 | 0.552075 |
| global_atomics_uniform | 1 | 8388352 | 5 | 101.615520 | 82549909.699 | 0.660399 |
| privatized_shared_hot_bin_90 | 1 | 8388352 | 5 | 9.529952 | 880209260.236 | 3.868647 |
| privatized_shared_mixed_hotset_75 | 1 | 8388352 | 5 | 28.369472 | 295682344.740 | 1.473298 |
| privatized_shared_uniform | 1 | 8388352 | 5 | 56.378592 | 148786120.803 | 0.971619 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114102Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114102Z_remaining_review_fixes.json).
- [histogram_atomic_contention_estimated_gbps.png](./results/charts/histogram_atomic_contention_estimated_gbps.png)
- [histogram_atomic_contention_median_gpu_ms.png](./results/charts/histogram_atomic_contention_median_gpu_ms.png)
- [histogram_atomic_contention_speedup_vs_global.png](./results/charts/histogram_atomic_contention_speedup_vs_global.png)
- [histogram_atomic_contention_stability_ratio.png](./results/charts/histogram_atomic_contention_stability_ratio.png)
- [histogram_atomic_contention_relative.csv](./results/tables/histogram_atomic_contention_relative.csv)
- [histogram_atomic_contention_stability.csv](./results/tables/histogram_atomic_contention_stability.csv)
- [histogram_atomic_contention_summary.csv](./results/tables/histogram_atomic_contention_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Histogram skew changes contention; shared replication and bank topology are not independently controlled.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
