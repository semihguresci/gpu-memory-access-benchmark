# Experiment 32 Results: Subgroup Stream Compaction Variants

## Run status and provenance

- Current export: **50/50 correctness checks passed**, 10 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `af4c3b692ceb986709762cc29ebd86ed08823b2abbfba17aca5368505f509e2f`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "32_subgroup_stream_compaction_variants", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\32_subgroup_stream_compaction_variants\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:44:23.908246Z` |
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
| gbps_mode | shader_payload_bytes |
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| payload_formula | (2*valid_count+block_count)*4 |
| subgroup_size | 32 |

## Key measurements

All cases share the reported **problem_size=2796032**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| shared_atomic_block_ratio_25 | 1 | 2796032 | 5 | 0.831904 | 3361003192.676 | 6.770772 |
| shared_atomic_block_ratio_5 | 1 | 2796032 | 5 | 0.547392 | 5107915351.339 | 2.124006 |
| shared_atomic_block_ratio_50 | 1 | 2796032 | 5 | 0.882016 | 3170046801.872 | 12.717230 |
| shared_atomic_block_ratio_75 | 1 | 2796032 | 5 | 0.934816 | 2990997158.799 | 17.992135 |
| shared_atomic_block_ratio_95 | 1 | 2796032 | 5 | 1.024288 | 2729732262.801 | 20.786951 |
| subgroup_ballot_ratio_25 | 1 | 2796032 | 5 | 0.861024 | 3247333407.663 | 6.541783 |
| subgroup_ballot_ratio_5 | 1 | 2796032 | 5 | 0.656640 | 4258089668.616 | 1.770626 |
| subgroup_ballot_ratio_50 | 1 | 2796032 | 5 | 0.930080 | 3006227421.297 | 12.060038 |
| subgroup_ballot_ratio_75 | 1 | 2796032 | 5 | 1.004608 | 2783206982.226 | 16.742188 |
| subgroup_ballot_ratio_95 | 1 | 2796032 | 5 | 1.104256 | 2532050539.005 | 19.281601 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114423Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114423Z_remaining_review_fixes.json).
- [subgroup_compaction_gpu_ms.png](./results/charts/subgroup_compaction_gpu_ms.png)
- [subgroup_compaction_speedup.png](./results/charts/subgroup_compaction_speedup.png)
- [subgroup_compaction_stability.png](./results/charts/subgroup_compaction_stability.png)
- [subgroup_stream_compaction_variants_relative.csv](./results/tables/subgroup_stream_compaction_variants_relative.csv)
- [subgroup_stream_compaction_variants_stability.csv](./results/tables/subgroup_stream_compaction_variants_stability.csv)
- [subgroup_stream_compaction_variants_summary.csv](./results/tables/subgroup_stream_compaction_variants_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Compaction scope is block-local. The subgroup path preserves logical order inside each block; shared atomic append is unordered. GB/s estimates survivor input/output and per-block count traffic, 4(2V+B) bytes; it excludes shared operations and is not measured DRAM traffic.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
