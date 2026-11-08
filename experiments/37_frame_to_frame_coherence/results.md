# Experiment 37 Results: Frame-to-Frame Coherence

## Run status and provenance

- Current export: **15/15 correctness checks passed**, 3 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `8a4ef1e3b973e5ce9e4cb262646c7d10b33cb48457373616754c09c65c3ee9ee`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "37_frame_to_frame_coherence", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\37_frame_to_frame_coherence\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:45:19.437771Z` |
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
| active_output_bytes | 29818880 |
| active_source_bytes | 3727360 |
| allocated_buffer_bytes | 33546240 |
| gbps_mode | shader_payload_bytes |
| gpu_time_scope | kernel_work_and_host_visibility_dependency |
| oracle | all_eight_frame_outputs_each_iteration |
| output_heap | 1 |
| output_memory_flags | 6 |
| output_memory_type | 3 |
| payload_formula | element_count*8_frames*(4_byte_read+4_byte_write) |
| resident_buffer_payload_bytes | 33546240 |
| scratch_size_bytes | 33554432 |
| source_heap | 1 |
| source_memory_flags | 6 |
| source_memory_type | 3 |

## Key measurements

All cases share the reported **problem_size=931840**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| block_scramble | 8 | 931840 | 5 | 2.723584 | 2737099351.443 | 21.896795 |
| coherent_shift | 8 | 931840 | 5 | 3.126784 | 2384149336.827 | 19.073195 |
| frame_random | 8 | 931840 | 5 | 3.763904 | 1980581863.937 | 15.844655 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114519Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114519Z_remaining_review_fixes.json).
- [frame_to_frame_coherence_ms.png](./results/charts/frame_to_frame_coherence_ms.png)
- [frame_to_frame_coherence_relative.png](./results/charts/frame_to_frame_coherence_relative.png)
- [frame_to_frame_coherence_stability.png](./results/charts/frame_to_frame_coherence_stability.png)
- [frame_to_frame_coherence_throughput.png](./results/charts/frame_to_frame_coherence_throughput.png)
- [frame_to_frame_coherence_relative.csv](./results/tables/frame_to_frame_coherence_relative.csv)
- [frame_to_frame_coherence_stability.csv](./results/tables/frame_to_frame_coherence_stability.csv)
- [frame_to_frame_coherence_summary.csv](./results/tables/frame_to_frame_coherence_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Every measured frame contributes to the validated output. Remapping changes spatial locality and collisions as well as temporal reuse, so timing does not isolate frame coherence.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
