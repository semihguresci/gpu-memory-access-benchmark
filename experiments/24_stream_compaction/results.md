# Experiment 24 Results: Stream Compaction

## Run status and provenance

- Current export: **50/50 correctness checks passed**, 10 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `1aaf1980491df6a73d84e0970e1be67656ace7287913c16989788d55745da8e5`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "24_stream_compaction", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\24_stream_compaction\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:41:15.788911Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `10` |
| row_count | `50` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| block_counts_span_bytes | 12288 |
| block_prefix_span_bytes | 12288 |
| counter_span_bytes | 4 |
| gbps_basis | estimated_source_level_storage_buffer_bytes |
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| input_span_bytes | 3145728 |
| local_prefix_span_bytes | 3145728 |
| output_span_bytes | 3145728 |
| pattern_seed | 604768804 |
| scratch_size_bytes | 33554432 |
| throughput_basis | input_elements_per_second |
| traffic_model | survivor_only_input_loads_includes_auxiliary_buffers |
| validation_mode | exact_stable_compaction, exact_unordered_compaction |

## Key measurements

All cases share the reported **problem_size=786432**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| global_atomic_append_ratio_25 | 1 | 786432 | 5 | 1.157952 | 679157685.293 | 2.717474 |
| global_atomic_append_ratio_5 | 1 | 786432 | 5 | 0.934944 | 841154122.600 | 0.672673 |
| global_atomic_append_ratio_50 | 1 | 786432 | 5 | 1.188736 | 661569936.470 | 5.287781 |
| global_atomic_append_ratio_75 | 1 | 786432 | 5 | 1.185920 | 663140852.671 | 7.949420 |
| global_atomic_append_ratio_95 | 1 | 786432 | 5 | 1.199712 | 655517324.158 | 9.960631 |
| three_stage_ratio_25 | 3 | 786432 | 5 | 0.628128 | 1252025064.955 | 10.096012 |
| three_stage_ratio_5 | 3 | 786432 | 5 | 0.442368 | 1777777777.778 | 8.643925 |
| three_stage_ratio_50 | 3 | 786432 | 5 | 0.665728 | 1181311286.291 | 14.241041 |
| three_stage_ratio_75 | 3 | 786432 | 5 | 0.696288 | 1129463670.205 | 18.127930 |
| three_stage_ratio_95 | 3 | 786432 | 5 | 0.715712 | 1098810694.805 | 21.160428 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114115Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114115Z_remaining_review_fixes.json).
- [stream_compaction_effective_gbps.png](./results/charts/stream_compaction_effective_gbps.png)
- [stream_compaction_median_gpu_ms.png](./results/charts/stream_compaction_median_gpu_ms.png)
- [stream_compaction_speedup_vs_atomic.png](./results/charts/stream_compaction_speedup_vs_atomic.png)
- [stream_compaction_stability_ratio.png](./results/charts/stream_compaction_stability_ratio.png)
- [stream_compaction_relative.csv](./results/tables/stream_compaction_relative.csv)
- [stream_compaction_stability.csv](./results/tables/stream_compaction_stability.csv)
- [stream_compaction_summary.csv](./results/tables/stream_compaction_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- The staged path preserves global input order; atomic append is unordered. GB/s estimates source-level storage-buffer accesses: atomic=16V bytes, staged=4(N+4V+4B+1) bytes, for N inputs, V survivors, B blocks. Atomic read-modify-write is modeled as 8 bytes; these are not measured DRAM transfers.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
