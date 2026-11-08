# Experiment 11 Results: Coalesced vs Strided Access

## Run status and provenance

- Current export: **35/35 correctness checks passed**, 7 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `8a10a509f1de6da93685eabb1eb1b6bf8d4e81fd8de494c7f560c7305c43cd6e`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "11_coalesced_vs_strided", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\11_coalesced_vs_strided\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:35:33.273925Z` |
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
| allocated_span_bytes | 1048576, 16777216, 2097152, 262144, 4194304, 524288, 8388608 |
| gpu_time_scope | dispatch_and_host_visibility |
| physical_span_bytes | 1048576, 16777216, 2097152, 262144, 4194304, 524288, 8388608 |
| validation_mode | exact_uint32 |

## Key measurements

All cases share the reported **problem_size=65536**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| stride_1 | 1 | 65536 | 5 | 0.037120 | 1765517241.379 | 14.124138 |
| stride_16 | 1 | 65536 | 5 | 0.432384 | 151568975.725 | 1.212552 |
| stride_2 | 1 | 65536 | 5 | 2.086880 | 31403818.140 | 0.251231 |
| stride_32 | 1 | 65536 | 5 | 0.519552 | 126139443.213 | 1.009116 |
| stride_4 | 1 | 65536 | 5 | 1.763968 | 37152601.408 | 0.297221 |
| stride_64 | 1 | 65536 | 5 | 0.520704 | 125860373.648 | 1.006883 |
| stride_8 | 1 | 65536 | 5 | 0.647392 | 101230784.440 | 0.809846 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113533Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113533Z_remaining_review_fixes.json).
- [coalesced_vs_strided_median_gbps.png](./results/charts/coalesced_vs_strided_median_gbps.png)
- [coalesced_vs_strided_median_gpu_ms.png](./results/charts/coalesced_vs_strided_median_gpu_ms.png)
- [coalesced_vs_strided_slowdown_vs_stride_1.png](./results/charts/coalesced_vs_strided_slowdown_vs_stride_1.png)
- [coalesced_vs_strided_stability_ratio.png](./results/charts/coalesced_vs_strided_stability_ratio.png)
- [coalesced_vs_strided_footprint.csv](./results/tables/coalesced_vs_strided_footprint.csv)
- [coalesced_vs_strided_relative.csv](./results/tables/coalesced_vs_strided_relative.csv)
- [coalesced_vs_strided_stability.csv](./results/tables/coalesced_vs_strided_stability.csv)
- [coalesced_vs_strided_summary.csv](./results/tables/coalesced_vs_strided_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Stride changes physical span as well as access locality; GB/s is a requested-byte measure.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
