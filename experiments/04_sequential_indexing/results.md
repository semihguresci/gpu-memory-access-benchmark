# Experiment 04 Results: Sequential Indexing

## Run status and provenance

- Current export: **520/520 correctness checks passed**, 1 variants, 13 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `1e021ab304271153b3a99e4c1f3ad5c49452824c666f9cb3da4ab31bfc7d93ce`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "04_sequential_indexing", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\04_sequential_indexing\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:32:46.612024Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `104` |
| row_count | `520` |
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
| sequential_read_write | 1 | 2796202 | 5 | 0.055872 | 50046570733.104 | 400.372566 |
| sequential_read_write | 4 | 2796202 | 5 | 0.221568 | 50480249855.575 | 403.841999 |
| sequential_read_write | 16 | 2796202 | 5 | 0.883584 | 50633818629.581 | 405.070549 |
| sequential_read_write | 64 | 2796202 | 5 | 3.570528 | 50120578245.010 | 400.964626 |
| sequential_read_write | 128 | 2796202 | 5 | 7.083232 | 50529737837.191 | 404.237903 |
| sequential_read_write | 256 | 2796202 | 5 | 14.988544 | 47758322089.190 | 382.066577 |
| sequential_read_write | 512 | 2796202 | 5 | 27.336864 | 52370872679.470 | 418.966981 |
| sequential_read_write | 1024 | 2796202 | 5 | 55.482240 | 51607700914.743 | 412.861607 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113246Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113246Z_remaining_review_fixes.json).
- [benchmark_summary.png](./results/charts/benchmark_summary.png)
- [sequential_indexing_gbps_vs_size.png](./results/charts/sequential_indexing_gbps_vs_size.png)
- [sequential_indexing_gpu_ms_vs_size.png](./results/charts/sequential_indexing_gpu_ms_vs_size.png)
- [sequential_indexing_multi_run_summary.csv](./results/tables/sequential_indexing_multi_run_summary.csv)
- [sequential_indexing_runs_index.csv](./results/tables/sequential_indexing_runs_index.csv)
- [sequential_indexing_status_overview.csv](./results/tables/sequential_indexing_status_overview.csv)
- [sequential_indexing_summary.csv](./results/tables/sequential_indexing_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Repeated dispatch timing includes compute dependencies and may reuse cached data.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
