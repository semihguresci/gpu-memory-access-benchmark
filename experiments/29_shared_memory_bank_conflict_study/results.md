# Experiment 29 Results: Shared Memory Bank Conflict Study

## Run status and provenance

- Current export: **35/35 correctness checks passed**, 7 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `2c3021b4f2f2fe5bc8c9cdd5377bc9e11dfce91179f1c7f37f58a07833b2e0b6`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "29_shared_memory_bank_conflict_study", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\29_shared_memory_bank_conflict_study\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:43:05.883654Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `7` |
| row_count | `35` |
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
| payload_bytes_per_element | 8 |
| scratch_size_bytes | 33554432 |
| shared_span_bytes | 1024, 16384, 2048, 32768, 33792, 4096, 8192 |

## Key measurements

All cases share the reported **problem_size=4194304**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| padded_fix | 1 | 4194304 | 5 | 1.486496 | 2821604632.639 | 22.572837 |
| stride_1 | 1 | 4194304 | 5 | 1.486336 | 2821908370.651 | 22.575267 |
| stride_16 | 1 | 4194304 | 5 | 1.493632 | 2808124089.468 | 22.464993 |
| stride_2 | 1 | 4194304 | 5 | 1.486656 | 2821300960.007 | 22.570408 |
| stride_32 | 1 | 4194304 | 5 | 1.911008 | 2194812371.272 | 17.558499 |
| stride_4 | 1 | 4194304 | 5 | 1.486880 | 2820875928.118 | 22.567007 |
| stride_8 | 1 | 4194304 | 5 | 1.487360 | 2819965576.592 | 22.559725 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114305Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114305Z_remaining_review_fixes.json).
- [bank_conflict_gpu_ms.png](./results/charts/bank_conflict_gpu_ms.png)
- [bank_conflict_slowdown.png](./results/charts/bank_conflict_slowdown.png)
- [bank_conflict_stability.png](./results/charts/bank_conflict_stability.png)
- [shared_memory_bank_conflict_study_relative.csv](./results/tables/shared_memory_bank_conflict_study_relative.csv)
- [shared_memory_bank_conflict_study_stability.csv](./results/tables/shared_memory_bank_conflict_study_stability.csv)
- [shared_memory_bank_conflict_study_summary.csv](./results/tables/shared_memory_bank_conflict_study_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Shared stride and padding are source access controls; hardware bank mapping and conflict counts are not measured.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
