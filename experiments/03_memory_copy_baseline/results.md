# Experiment 03 Results: Memory Copy Baseline

## Run status and provenance

- Current export: **75/75 correctness checks passed**, 3 variants, 5 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `a28dd677e6c10b4b77e35766f4877772e67c80d449e385c4849985d39ac8c3ee`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "03_memory_copy_baseline", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\03_memory_copy_baseline\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:32:18.349826Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `15` |
| row_count | `75` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| readback_scope | copy_and_host_visibility |

## Key measurements

All cases share the reported **problem_size=2796202**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| read_only | 1 | 2796202 | 5 | 0.041536 | 67319963405.239 | 269.279854 |
| read_write_copy | 1 | 2796202 | 5 | 0.057056 | 49008027201.346 | 392.064218 |
| write_only | 1 | 2796202 | 5 | 0.032096 | 87119952642.074 | 348.479811 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113218Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113218Z_remaining_review_fixes.json).
- [benchmark_summary.png](./results/charts/benchmark_summary.png)
- [memory_copy_baseline_gbps_vs_size.png](./results/charts/memory_copy_baseline_gbps_vs_size.png)
- [memory_copy_baseline_gpu_ms_vs_size.png](./results/charts/memory_copy_baseline_gpu_ms_vs_size.png)
- [memory_copy_baseline_multi_run_summary.csv](./results/tables/memory_copy_baseline_multi_run_summary.csv)
- [memory_copy_baseline_runs_index.csv](./results/tables/memory_copy_baseline_runs_index.csv)
- [memory_copy_baseline_status_overview.csv](./results/tables/memory_copy_baseline_status_overview.csv)
- [memory_copy_baseline_summary.csv](./results/tables/memory_copy_baseline_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Read-only correctness observes configured probes; requested bytes are not measured external-memory traffic.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
