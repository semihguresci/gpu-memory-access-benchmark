# Experiment 17 Results: Tile Size Sweep

## Run status and provenance

- Current export: **40/40 correctness checks passed**, 8 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `9573309ca0557519a37cabf4f1eafc4f7031f36a987db95b66cd8e094d01a2f6`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "17_tile_size_sweep", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\17_tile_size_sweep\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:38:31.793652Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `8` |
| row_count | `40` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| gpu_time_scope | dispatch_and_host_visibility |
| scratch_size_bytes | 33554432 |
| validation_mode | exact_uint32 |

## Key measurements

All cases share the reported **problem_size=4194240**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| direct_global_t16 | 1 | 4194240 | 5 | 1.806656 | 2321548761.912 | 315.730632 |
| direct_global_t32 | 1 | 4194240 | 5 | 1.505248 | 2786411275.750 | 378.951934 |
| direct_global_t64 | 1 | 4194240 | 5 | 1.505024 | 2786825990.815 | 379.008335 |
| direct_global_t8 | 1 | 4194240 | 5 | 3.122176 | 1343370777.304 | 182.698426 |
| shared_tiled_t16 | 1 | 4194240 | 5 | 2.007552 | 2089231063.504 | 33.427697 |
| shared_tiled_t32 | 1 | 4194240 | 5 | 1.489568 | 2815742550.860 | 33.788911 |
| shared_tiled_t64 | 1 | 4194240 | 5 | 1.501920 | 2792585490.572 | 27.925855 |
| shared_tiled_t8 | 1 | 4194240 | 5 | 2.732928 | 1534705634.397 | 36.832935 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113831Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113831Z_remaining_review_fixes.json).
- [tile_size_sweep_estimated_gbps.png](./results/charts/tile_size_sweep_estimated_gbps.png)
- [tile_size_sweep_median_gpu_ms.png](./results/charts/tile_size_sweep_median_gpu_ms.png)
- [tile_size_sweep_speedup_vs_direct.png](./results/charts/tile_size_sweep_speedup_vs_direct.png)
- [tile_size_sweep_stability_ratio.png](./results/charts/tile_size_sweep_stability_ratio.png)
- [tile_size_sweep_relative.csv](./results/tables/tile_size_sweep_relative.csv)
- [tile_size_sweep_stability.csv](./results/tables/tile_size_sweep_stability.csv)
- [tile_size_sweep_summary.csv](./results/tables/tile_size_sweep_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Tile size and local workgroup size vary together; timing does not isolate either factor.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
