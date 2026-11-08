# Experiment 18 Results: Register Pressure Proxy

## Run status and provenance

- Current export: **20/20 correctness checks passed**, 4 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `87715ed042cc94bf789b1800d77c80ed91b55dddcfc9bde306404633541abee8`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "18_register_pressure_proxy", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\18_register_pressure_proxy\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:38:54.079524Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `4` |
| row_count | `20` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| destination_span_bytes | 16777216 |
| gpu_time_scope | dispatch_and_host_visibility |
| independent_chain_count | 32 |
| output_contract | identical_across_variants |
| payload_bytes_per_element | 8 |
| source_span_bytes | 16777216 |
| validation_mode | exact_uint32 |

## Key measurements

All cases share the reported **problem_size=4194304**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| temp_16 | 1 | 4194304 | 5 | 1.498240 | 2799487398.548 | 22.395899 |
| temp_32 | 1 | 4194304 | 5 | 1.793632 | 2338441776.239 | 18.707534 |
| temp_4 | 1 | 4194304 | 5 | 1.523680 | 2752745983.409 | 22.021968 |
| temp_8 | 1 | 4194304 | 5 | 1.532768 | 2736424560.012 | 21.891396 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113854Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113854Z_remaining_review_fixes.json).
- [register_pressure_proxy_median_gpu_ms.png](./results/charts/register_pressure_proxy_median_gpu_ms.png)
- [register_pressure_proxy_speedup_vs_baseline.png](./results/charts/register_pressure_proxy_speedup_vs_baseline.png)
- [register_pressure_proxy_stability_ratio.png](./results/charts/register_pressure_proxy_stability_ratio.png)
- [register_pressure_proxy_relative.csv](./results/tables/register_pressure_proxy_relative.csv)
- [register_pressure_proxy_stability.csv](./results/tables/register_pressure_proxy_stability.csv)
- [register_pressure_proxy_summary.csv](./results/tables/register_pressure_proxy_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- All variants execute the same 32 chains and 256 updates in batches of 4/8/16/32 live states. Source batch width does not prove register allocation, spills, or occupancy; compiler and scheduling effects remain.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
