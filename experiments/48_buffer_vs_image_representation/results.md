# Experiment 48 Results: Buffer Vs Image Representation

## Run status and provenance

- Current export: **200/200 correctness checks passed**, 8 variants, 5 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `286ed94d89855a0e5b714cbe48066d49b1e17b499668197bae0bc9f6ca8cbadc`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "48_buffer_vs_image_representation", "--iterations", "5", "--warmup", "2", "--size", "128M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\48_buffer_vs_image_representation\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:48:55.497223Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `40` |
| row_count | `200` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| allocated_image_bytes | 19398656 |
| buffer_span_bytes | 1048576, 16777216, 19173376, 262144, 4194304 |
| estimated_request_bytes | 1310720, 20971520, 5242880, 83886080, 95866880 |
| gbps_mode | estimated_shader_request_bytes |
| grid_height | 1024, 256, 4096, 4681, 64 |
| grid_width | 1024 |
| image_heap_index | 0 |
| image_memory_type_index | 1 |
| live_buffer_count | 7 |
| oracle | reference_complement_before_every_dispatch |
| output_preparation | transfer_outside_dispatch_timing |
| scratch_size_bytes | 134217728 |

## Key measurements

All cases share the reported **problem_size=4793344**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| column_major_buffer_neighborhood_clamped | 1 | 4793344 | 5 | 0.976864 | 4906869328.791 | 98.137387 |
| column_major_buffer_random_scatter | 1 | 4793344 | 5 | 6.221792 | 770412125.638 | 15.408243 |
| row_major_buffer_neighborhood_clamped | 1 | 4793344 | 5 | 0.154208 | 31083627308.570 | 621.672546 |
| row_major_buffer_random_scatter | 1 | 4793344 | 5 | 5.960832 | 804140093.195 | 16.082802 |
| sampled_image_neighborhood_clamped | 1 | 4793344 | 5 | 0.179584 | 26691375623.664 | 533.827512 |
| sampled_image_random_scatter | 1 | 4793344 | 5 | 6.193440 | 773938877.264 | 15.478778 |
| storage_image_neighborhood_clamped | 1 | 4793344 | 5 | 0.178816 | 26806012884.753 | 536.120258 |
| storage_image_random_scatter | 1 | 4793344 | 5 | 6.207008 | 772247111.652 | 15.444942 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114855Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114855Z_remaining_review_fixes.json).
- [buffer_vs_image_representation_ms_vs_size.png](./results/charts/buffer_vs_image_representation_ms_vs_size.png)
- [buffer_vs_image_representation_ratios_neighborhood.png](./results/charts/buffer_vs_image_representation_ratios_neighborhood.png)
- [buffer_vs_image_representation_ratios_random.png](./results/charts/buffer_vs_image_representation_ratios_random.png)
- [buffer_vs_image_representation_stability.png](./results/charts/buffer_vs_image_representation_stability.png)
- [buffer_vs_image_representation_checksums.csv](./results/tables/buffer_vs_image_representation_checksums.csv)
- [buffer_vs_image_representation_relative.csv](./results/tables/buffer_vs_image_representation_relative.csv)
- [buffer_vs_image_representation_representation_effects.csv](./results/tables/buffer_vs_image_representation_representation_effects.csv)
- [buffer_vs_image_representation_stability.csv](./results/tables/buffer_vs_image_representation_stability.csv)
- [buffer_vs_image_representation_summary.csv](./results/tables/buffer_vs_image_representation_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Both image axes are bounded by the device image limit. Results cover only R32_UINT, exact unfiltered fetches, and the implemented shapes; requested bytes are not measured physical memory traffic.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
