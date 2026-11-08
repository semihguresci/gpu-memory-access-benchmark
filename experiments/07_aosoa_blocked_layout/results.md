# Experiment 07 Results: AoSoA Blocked Layout

## Run status and provenance

- Current export: **60/60 correctness checks passed**, 6 variants, 2 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `4147af1073ca1af273c1ea9cbb80f5fddc67c6c4dff5b427aaba58d3100924d9`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "07_aosoa_blocked_layout", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\07_aosoa_blocked_layout\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:34:40.558947Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `12` |
| row_count | `60` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| first_buffer_payload_bytes | 11184128, 699008 |
| gbps_basis | useful_field_payload |
| gpu_time_scope | dispatch_and_host_visibility |
| memory_heap_index | 1 |
| memory_property_flags | 6 |
| memory_type_index | 3 |

## Key measurements

All cases share the reported **problem_size=174752**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| aos | 1 | 174752 | 5 | 5.010432 | 34877631.310 | 2.790211 |
| aosoa_b16 | 1 | 174752 | 5 | 0.868864 | 201126988.804 | 16.090159 |
| aosoa_b32 | 1 | 174752 | 5 | 0.854400 | 204531835.206 | 16.362547 |
| aosoa_b4 | 1 | 174752 | 5 | 1.917376 | 91141226.343 | 7.291298 |
| aosoa_b8 | 1 | 174752 | 5 | 1.422368 | 122859906.860 | 9.828793 |
| soa | 1 | 174752 | 5 | 0.820512 | 212979212.979 | 17.038337 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_113440Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_113440Z_remaining_review_fixes.json).
- [aosoa_blocked_layout_gbps_vs_size.png](./results/charts/aosoa_blocked_layout_gbps_vs_size.png)
- [aosoa_blocked_layout_gpu_ms_vs_size.png](./results/charts/aosoa_blocked_layout_gpu_ms_vs_size.png)
- [benchmark_summary.png](./results/charts/benchmark_summary.png)
- [aosoa_blocked_layout_multi_run_summary.csv](./results/tables/aosoa_blocked_layout_multi_run_summary.csv)
- [aosoa_blocked_layout_runs_index.csv](./results/tables/aosoa_blocked_layout_runs_index.csv)
- [aosoa_blocked_layout_status_overview.csv](./results/tables/aosoa_blocked_layout_status_overview.csv)
- [aosoa_blocked_layout_summary.csv](./results/tables/aosoa_blocked_layout_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Active field operations are matched; blocking and physical transaction effects are not independently observed.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
