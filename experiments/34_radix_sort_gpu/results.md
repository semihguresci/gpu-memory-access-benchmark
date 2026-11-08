# Experiment 34 Results: Radix Sort on GPU

## Run status and provenance

- Current export: **50/50 correctness checks passed**, 2 variants, 5 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `7ec40ff123207ed779e44dca8399949f15e32239a605a53267ba22fd0245828d`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "34_radix_sort_gpu", "--iterations", "5", "--warmup", "2", "--size", "128M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\34_radix_sort_gpu\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:45:00.645136Z` |
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
| gbps_mode | estimated_global_bytes |
| gpu_time_scope | kernel_work_and_host_visibility_dependency |

## Key measurements

All cases share the reported **problem_size=4194304**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| 4bit_8pass | 24 | 4194304 | 5 | 91.081216 | 46050153.744 | 7.644331 |
| 8bit_4pass | 12 | 4194304 | 5 | 240.314720 | 17453379.468 | 2.234050 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114500Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114500Z_remaining_review_fixes.json).
- [radix_sort_gpu_ms.png](./results/charts/radix_sort_gpu_ms.png)
- [radix_sort_stability.png](./results/charts/radix_sort_stability.png)
- [radix_sort_throughput.png](./results/charts/radix_sort_throughput.png)
- [radix_sort_gpu_relative.csv](./results/tables/radix_sort_gpu_relative.csv)
- [radix_sort_gpu_stability.csv](./results/tables/radix_sort_gpu_stability.csv)
- [radix_sort_gpu_summary.csv](./results/tables/radix_sort_gpu_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- This is stable key-only radix sorting; radix width changes pass count and auxiliary traffic, with dispatch count recorded per complete sort.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
