# Experiment 05 Results: Global ID Mapping Variants

## Run status and provenance

- Current export: **1560/1560 correctness checks passed**, 3 variants, 13 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `be08956968e5cd1ca95c42e83596740409057560bb04eb420ae05bcb4b0dfb65`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "05_global_id_mapping_variants", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\05_global_id_mapping_variants\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:34:08.798223Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `312` |
| row_count | `1560` |
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
| direct | 1 | 2796202 | 5 | 0.053664 | 52105731961.837 | 416.845856 |
| direct | 4 | 2796202 | 5 | 0.222976 | 50161488231.917 | 401.291906 |
| direct | 16 | 2796202 | 5 | 0.883680 | 50628317943.147 | 405.026544 |
| direct | 64 | 2796202 | 5 | 3.547776 | 50442003102.789 | 403.536025 |
| direct | 128 | 2796202 | 5 | 7.088064 | 50495291238.905 | 403.962330 |
| direct | 256 | 2796202 | 5 | 14.746848 | 48541065317.822 | 388.328523 |
| direct | 512 | 2796202 | 5 | 27.407968 | 52235007863.407 | 417.880063 |
| direct | 1024 | 2796202 | 5 | 54.919424 | 52136578271.469 | 417.092626 |
| grid_stride | 1 | 2796202 | 5 | 0.054336 | 51461314782.097 | 411.690518 |
| grid_stride | 4 | 2796202 | 5 | 0.223616 | 50017923583.286 | 400.143389 |
| grid_stride | 16 | 2796202 | 5 | 0.889920 | 50273318950.018 | 402.186552 |
| grid_stride | 64 | 2796202 | 5 | 4.419904 | 40488872156.499 | 323.910977 |
| grid_stride | 128 | 2796202 | 5 | 7.142080 | 50113392177.069 | 400.907137 |
| grid_stride | 256 | 2796202 | 5 | 14.442752 | 49563110409.983 | 396.504883 |
| grid_stride | 512 | 2796202 | 5 | 27.686688 | 51709161601.417 | 413.673293 |
| grid_stride | 1024 | 2796202 | 5 | 54.885984 | 52168343160.250 | 417.346745 |
| offset | 1 | 2796202 | 5 | 0.054240 | 51552396755.162 | 412.419174 |
| offset | 4 | 2796202 | 5 | 0.226272 | 49430808937.915 | 395.446472 |
| offset | 16 | 2796202 | 5 | 0.903072 | 49541157294.214 | 396.329258 |
| offset | 64 | 2796202 | 5 | 4.443232 | 40276296173.596 | 322.210369 |
| offset | 128 | 2796202 | 5 | 7.210496 | 49637896754.953 | 397.103174 |
| offset | 256 | 2796202 | 5 | 14.406016 | 49689498609.470 | 397.515989 |
| offset | 512 | 2796202 | 5 | 28.144320 | 50868360791.805 | 406.946886 |
| offset | 1024 | 2796202 | 5 | 55.047936 | 52014862973.246 | 416.118904 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113408Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113408Z_remaining_review_fixes.json).
- [benchmark_summary.png](./results/charts/benchmark_summary.png)
- [global_id_mapping_variants_gbps_vs_size.png](./results/charts/global_id_mapping_variants_gbps_vs_size.png)
- [global_id_mapping_variants_gpu_ms_vs_size.png](./results/charts/global_id_mapping_variants_gpu_ms_vs_size.png)
- [global_id_mapping_variants_multi_run_summary.csv](./results/tables/global_id_mapping_variants_multi_run_summary.csv)
- [global_id_mapping_variants_runs_index.csv](./results/tables/global_id_mapping_variants_runs_index.csv)
- [global_id_mapping_variants_status_overview.csv](./results/tables/global_id_mapping_variants_status_overview.csv)
- [global_id_mapping_variants_summary.csv](./results/tables/global_id_mapping_variants_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Index mappings also change invocation scheduling; repeated dispatches include compute dependencies.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
