# Experiment 47 Results: Block Reorder Factorial

## Run status and provenance

- Current export: **560/560 correctness checks passed**, 28 variants, 4 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `f33786046cb0bb9613e8a2f350890be745936d508647dd3b2e6549a32a44004d`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "47_block_reorder_factorial", "--iterations", "5", "--warmup", "2", "--size", "128M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\47_block_reorder_factorial\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:47:59.032164Z` |
| generator | `gpu_memory_layout_experiments` |
| gpu_device_uuid | `da7acb583a2f9c0e88f36758750c1468` |
| gpu_name | `NVIDIA GeForce RTX 2080 SUPER` |
| gpu_timestamps_supported | `true` |
| measurement_revision | `2026-11-08-remaining-review-fixes` |
| provenance | `{"binary_sha256": "886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120", "git_revision": "262350e79e975b57082b15b324a8365dd8564897", "shader_bundle_file_count": 107, "shader_bundle_sha256": "dc82832bcd22dcbd2d2f9208d0112a92fc722922eec5119138669bd80b4cff1e", "source_tree_sha256": "799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4", "working_tree_dirty": true}` |
| result_count | `112` |
| row_count | `560` |
| shader_binary_directory | `F:/Projects/gpu-memory-layout-playground/build-tests-vs/shaders` |
| timed_iterations | `5` |
| validation_enabled | `false` |
| vulkan_api_version | `1.4.341` |
| warmup_iterations | `2` |
| schema | `{"name": "gpu_memory_layout_results", "version": "1.2.0"}` |

Exported configuration and metric definitions (all distinct values):

| Row-note field | Recorded values |
|---|---|
| block_mode | sequential, shuffled_by_block_perm |
| block_perm_seed | 1205982656 |
| block_perm_span_bytes | 1024, 104832, 13104, 16384, 2048, 209664, 256, 26208, 32768, 4096, 512, 52416, 65536, 8192 |
| block_width | 1024, 128, 256, 512, 64 |
| buffer_span_bytes | 1048576, 13418496, 262144, 4194304 |
| estimated_request_bytes | 12582912, 3145728, 40255488, 786432 |
| eviction_buffer_bytes | 67108864 |
| gbps_mode | estimated_shader_request_bytes |
| live_allocation_bytes | 120992512 |
| live_block_perm_span_bytes | 209664 |
| live_buffer_count | 6 |
| live_payload_bytes | 120992512 |
| oracle | reference_complement_before_every_dispatch |
| output_preparation | transfer_after_residency_setup_outside_dispatch_timing |
| output_preparation_cache_effect | output_copy_may_perturb_cache_state |
| residency | evicted_buffer_sweep, warm |
| scratch_budget_scope | live_buffer_payload |
| scratch_size_bytes | 134217728 |
| source_pattern_seed | 2654435769 |
| source_span_bytes | 1048576, 13418496, 262144, 4194304 |
| subgroup_size | 32 |
| warm_residency_setup | unmeasured_main_dispatch_before_output_poison |
| within_mode | bit_reversal, ordered |
| workload_buffer_span_bytes | 13418496 |

## Key measurements

All cases share the reported **problem_size=3354624**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| w1024_blocks_seq_within_ordered | 1 | 3354624 | 5 | 0.103776 | 32325624421.832 | 387.907493 |
| w1024_blocks_seq_within_ordered_evicted | 1 | 3354624 | 5 | 0.103904 | 32285802279.027 | 387.429627 |
| w1024_blocks_seq_within_reversed | 1 | 3354624 | 5 | 0.147840 | 22690909090.909 | 272.290909 |
| w1024_blocks_seq_within_reversed_evicted | 1 | 3354624 | 5 | 0.147296 | 22774712144.254 | 273.296546 |
| w1024_blocks_shuffled_within_ordered | 1 | 3354624 | 5 | 0.103296 | 32475836431.227 | 389.710037 |
| w1024_blocks_shuffled_within_ordered_evicted | 1 | 3354624 | 5 | 0.103936 | 32275862068.966 | 387.310345 |
| w1024_blocks_shuffled_within_reversed | 1 | 3354624 | 5 | 0.146656 | 22874099934.541 | 274.489199 |
| w1024_blocks_shuffled_within_reversed_evicted | 1 | 3354624 | 5 | 0.146528 | 22894081677.222 | 274.728980 |
| w128_blocks_seq_within_ordered | 1 | 3354624 | 5 | 0.108256 | 30987880579.367 | 371.854567 |
| w128_blocks_seq_within_reversed | 1 | 3354624 | 5 | 0.110592 | 30333333333.333 | 364.000000 |
| w128_blocks_shuffled_within_ordered | 1 | 3354624 | 5 | 0.109152 | 30733509234.828 | 368.802111 |
| w128_blocks_shuffled_within_reversed | 1 | 3354624 | 5 | 0.110080 | 30474418604.651 | 365.693023 |
| w256_blocks_seq_within_ordered | 1 | 3354624 | 5 | 0.107328 | 31255813953.488 | 375.069767 |
| w256_blocks_seq_within_reversed | 1 | 3354624 | 5 | 0.112288 | 29875178113.423 | 358.502137 |
| w256_blocks_shuffled_within_ordered | 1 | 3354624 | 5 | 0.107904 | 31088967971.530 | 373.067616 |
| w256_blocks_shuffled_within_reversed | 1 | 3354624 | 5 | 0.111776 | 30012024048.096 | 360.144289 |
| w512_blocks_seq_within_ordered | 1 | 3354624 | 5 | 0.106432 | 31518941671.678 | 378.227300 |
| w512_blocks_seq_within_reversed | 1 | 3354624 | 5 | 0.121120 | 27696697490.092 | 332.360370 |
| w512_blocks_shuffled_within_ordered | 1 | 3354624 | 5 | 0.105792 | 31709618874.773 | 380.515426 |
| w512_blocks_shuffled_within_reversed | 1 | 3354624 | 5 | 0.121216 | 27674762407.603 | 332.097149 |
| w64_blocks_seq_within_ordered | 1 | 3354624 | 5 | 0.109088 | 30751540041.068 | 369.018480 |
| w64_blocks_seq_within_ordered_evicted | 1 | 3354624 | 5 | 0.109184 | 30724501758.499 | 368.694021 |
| w64_blocks_seq_within_reversed | 1 | 3354624 | 5 | 0.109280 | 30697510980.966 | 368.370132 |
| w64_blocks_seq_within_reversed_evicted | 1 | 3354624 | 5 | 0.109440 | 30652631578.947 | 367.831579 |
| w64_blocks_shuffled_within_ordered | 1 | 3354624 | 5 | 0.115008 | 29168614357.262 | 350.023372 |
| w64_blocks_shuffled_within_ordered_evicted | 1 | 3354624 | 5 | 0.115200 | 29120000000.000 | 349.440000 |
| w64_blocks_shuffled_within_reversed | 1 | 3354624 | 5 | 0.115104 | 29144286905.755 | 349.731443 |
| w64_blocks_shuffled_within_reversed_evicted | 1 | 3354624 | 5 | 0.115648 | 29007194244.604 | 348.086331 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114759Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114759Z_remaining_review_fixes.json).
- [block_reorder_factorial_effects_by_width.png](./results/charts/block_reorder_factorial_effects_by_width.png)
- [block_reorder_factorial_ms_vs_size_smallest_width.png](./results/charts/block_reorder_factorial_ms_vs_size_smallest_width.png)
- [block_reorder_factorial_residency_ratio.png](./results/charts/block_reorder_factorial_residency_ratio.png)
- [block_reorder_factorial_stability.png](./results/charts/block_reorder_factorial_stability.png)
- [block_reorder_factorial_checksums.csv](./results/tables/block_reorder_factorial_checksums.csv)
- [block_reorder_factorial_effects.csv](./results/tables/block_reorder_factorial_effects.csv)
- [block_reorder_factorial_relative.csv](./results/tables/block_reorder_factorial_relative.csv)
- [block_reorder_factorial_residency.csv](./results/tables/block_reorder_factorial_residency.csv)
- [block_reorder_factorial_stability.csv](./results/tables/block_reorder_factorial_stability.csv)
- [block_reorder_factorial_summary.csv](./results/tables/block_reorder_factorial_summary.csv)
- Omitted 1 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- All six live buffer payloads, including the fixed 64 MiB eviction buffer, fit within the CLI scratch budget; allocation padding is recorded separately. Warm cases use an unmeasured priming dispatch; evicted cases use the sweep. Both then upload reference-complement outputs before the measured dispatch, so stale results cannot pass. That common output copy and the sweep perturb cache state; neither certifies input residency. Block widths exceed the recorded device subgroup size; identity/shuffled tables have equal requested bytes. Transaction counts are not measured.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
