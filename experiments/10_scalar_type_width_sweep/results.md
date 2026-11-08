# Experiment 10 Results: Scalar Type Width Sweep

## Run status and provenance

- Current export: **50/50 correctness checks passed**, 5 variants, 2 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `bab8e0e95b65728900ae773e45fcd2dc0ae97bbee631b8b8e5298ef9c7e028c7`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "10_scalar_type_width_sweep", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\10_scalar_type_width_sweep\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:35:22.327728Z` |
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
| gpu_time_scope | dispatch_and_host_visibility |

## Key measurements

All cases share the reported **problem_size=2581110**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| fp16_storage | 1 | 2581110 | 5 | 0.470944 | 5480715329.211 | 21.922861 |
| fp32 | 1 | 2581110 | 5 | 0.926368 | 2786268523.956 | 22.290148 |
| u16 | 1 | 2581110 | 5 | 0.471328 | 5476250084.867 | 21.905000 |
| u32 | 1 | 2581110 | 5 | 0.926304 | 2786461032.231 | 22.291688 |
| u8 | 1 | 2581110 | 5 | 0.243136 | 10615910436.957 | 21.231821 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113522Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113522Z_remaining_review_fixes.json).
- [scalar_type_width_sweep_error_metrics.png](./results/charts/scalar_type_width_sweep_error_metrics.png)
- [scalar_type_width_sweep_error_vs_fp32.png](./results/charts/scalar_type_width_sweep_error_vs_fp32.png)
- [scalar_type_width_sweep_gpu_ms_per_million_elements.png](./results/charts/scalar_type_width_sweep_gpu_ms_per_million_elements.png)
- [scalar_type_width_sweep_median_gbps.png](./results/charts/scalar_type_width_sweep_median_gbps.png)
- [scalar_type_width_sweep_median_gpu_ms.png](./results/charts/scalar_type_width_sweep_median_gpu_ms.png)
- [scalar_type_width_sweep_p95_to_median_ratio.png](./results/charts/scalar_type_width_sweep_p95_to_median_ratio.png)
- [scalar_type_width_sweep_speedup_vs_fp32.png](./results/charts/scalar_type_width_sweep_speedup_vs_fp32.png)
- [scalar_type_width_sweep_largest_size_comparison.csv](./results/tables/scalar_type_width_sweep_largest_size_comparison.csv)
- [scalar_type_width_sweep_scaling_by_size.csv](./results/tables/scalar_type_width_sweep_scaling_by_size.csv)
- [scalar_type_width_sweep_speedup_crossover.csv](./results/tables/scalar_type_width_sweep_speedup_crossover.csv)
- [scalar_type_width_sweep_speedup_vs_fp32_by_size.csv](./results/tables/scalar_type_width_sweep_speedup_vs_fp32_by_size.csv)
- [scalar_type_width_sweep_stability_overview.csv](./results/tables/scalar_type_width_sweep_stability_overview.csv)
- [scalar_type_width_sweep_status_overview.csv](./results/tables/scalar_type_width_sweep_status_overview.csv)
- [scalar_type_width_sweep_summary.csv](./results/tables/scalar_type_width_sweep_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Manual packing changes transport and conversion cost. Format-specific correctness residuals and separately reported max/RMS error versus a common fp32 oracle answer different questions; native narrow arithmetic is not covered.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
