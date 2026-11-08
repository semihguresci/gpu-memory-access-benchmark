# Experiment 27 Results: Cache Thrashing, Random vs Sequential

## Run status and provenance

- Current export: **15/15 correctness checks passed**, 3 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `4aa328a34624ce43295ce4eef120de91cf3c2dd5d5665ad124f24811a0f2a24a`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "27_cache_thrashing_random_vs_sequential", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\27_cache_thrashing_random_vs_sequential\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:42:06.073116Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `3` |
| row_count | `15` |
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
| payload_bytes_per_element | 12 |
| seed | 324508639 |

## Key measurements

All cases share the reported **problem_size=2796032**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| block_shuffled | 1 | 2796032 | 5 | 1.725664 | 1620264431.546 | 19.443173 |
| random | 1 | 2796032 | 5 | 10.061312 | 277899343.545 | 3.334792 |
| sequential | 1 | 2796032 | 5 | 1.731520 | 1614784697.838 | 19.377416 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114206Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114206Z_remaining_review_fixes.json).
- [cache_thrashing_random_vs_sequential_estimated_gbps.png](./results/charts/cache_thrashing_random_vs_sequential_estimated_gbps.png)
- [cache_thrashing_random_vs_sequential_median_gpu_ms.png](./results/charts/cache_thrashing_random_vs_sequential_median_gpu_ms.png)
- [cache_thrashing_random_vs_sequential_slowdown_vs_sequential.png](./results/charts/cache_thrashing_random_vs_sequential_slowdown_vs_sequential.png)
- [cache_thrashing_random_vs_sequential_stability_ratio.png](./results/charts/cache_thrashing_random_vs_sequential_stability_ratio.png)
- [cache_thrashing_random_vs_sequential_relative.csv](./results/tables/cache_thrashing_random_vs_sequential_relative.csv)
- [cache_thrashing_random_vs_sequential_stability.csv](./results/tables/cache_thrashing_random_vs_sequential_stability.csv)
- [cache_thrashing_random_vs_sequential_summary.csv](./results/tables/cache_thrashing_random_vs_sequential_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Permutation changes scheduling locality; within-subgroup coalescing and cache reuse are not independently varied.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
