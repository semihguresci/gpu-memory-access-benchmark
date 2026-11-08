# Experiment 20 Results: Barrier and Synchronization Cost

## Run status and provenance

- Current export: **50/50 correctness checks passed**, 10 variants, 1 problem sizes.
- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.
- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.
- Raw export SHA-256: `7570b1b95e56b8d84158f25f9898372be00e0bdfe48f202c4aa55ff7eea4088a`.

| Runtime metadata | Recorded value |
|---|---|
| command_line | `["F:\\Projects\\gpu-memory-layout-playground\\build-tests-vs\\Release\\gpu_memory_layout_experiments.exe", "--experiment", "20_barrier_synchronization_cost", "--iterations", "5", "--warmup", "2", "--size", "32M", "--output", "F:\\Projects\\gpu-memory-layout-playground\\experiments\\20_barrier_synchronization_cost\\results\\tables\\benchmark_results.json"]` |
| driver_version | `2559967232` |
| exported_at_utc | `2026-11-08T11:40:16.345226Z` |
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
| output_contract | identical_across_variants |
| scratch_size_bytes | 33554432 |
| shared_seed_reads_per_invocation | 8 |
| shared_seed_writes_per_workgroup | 1 |
| validation_mode | exact_uint32 |

## Key measurements

All cases share the reported **problem_size=4194304**, the largest size common to every variant and dispatch-count case.

Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.

| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |
|---|---:|---:|---:|---:|---:|---:|
| flat_loop_b0 | 1 | 4194304 | 5 | 1.502624 | 2791319718.040 | 22.330558 |
| flat_loop_b1 | 1 | 4194304 | 5 | 1.500096 | 2796023721.149 | 22.368190 |
| flat_loop_b2 | 1 | 4194304 | 5 | 1.500320 | 2795606270.662 | 22.364850 |
| flat_loop_b4 | 1 | 4194304 | 5 | 1.498080 | 2799786393.250 | 22.398291 |
| flat_loop_b8 | 1 | 4194304 | 5 | 1.499360 | 2797396222.388 | 22.379170 |
| tiled_regions_b0 | 1 | 4194304 | 5 | 1.499168 | 2797754487.823 | 22.382036 |
| tiled_regions_b1 | 1 | 4194304 | 5 | 1.498976 | 2798112845.036 | 22.384903 |
| tiled_regions_b2 | 1 | 4194304 | 5 | 1.500544 | 2795188944.809 | 22.361512 |
| tiled_regions_b4 | 1 | 4194304 | 5 | 1.501440 | 2793520886.616 | 22.348167 |
| tiled_regions_b8 | 1 | 4194304 | 5 | 1.498752 | 2798531044.496 | 22.388248 |

## Artifacts

- Current raw data: [benchmark_results.json](./results/tables/benchmark_results.json).
- Archives with exactly matching JSON content: [20261108_114016Z_remaining_review_fixes.json](./runs/nvidia_geforce_rtx_2080_super/20261108_114016Z_remaining_review_fixes.json).
- [barrier_synchronization_cost_estimated_gbps.png](./results/charts/barrier_synchronization_cost_estimated_gbps.png)
- [barrier_synchronization_cost_median_gpu_ms.png](./results/charts/barrier_synchronization_cost_median_gpu_ms.png)
- [barrier_synchronization_cost_speedup_vs_flat.png](./results/charts/barrier_synchronization_cost_speedup_vs_flat.png)
- [barrier_synchronization_cost_stability_ratio.png](./results/charts/barrier_synchronization_cost_stability_ratio.png)
- [barrier_synchronization_cost_relative.csv](./results/tables/barrier_synchronization_cost_relative.csv)
- [barrier_synchronization_cost_stability.csv](./results/tables/barrier_synchronization_cost_stability.csv)
- [barrier_synchronization_cost_summary.csv](./results/tables/barrier_synchronization_cost_summary.csv)
- Omitted 0 unbound CSV/chart files. Linked derived outputs are bound by SHA-256 to verified analyzer/plot code and sources including the exact current raw export.

## Interpretation and limitations

- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.
- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.
- Every variant performs the same eight work phases and publishes one immutable shared seed. b0 has one mandatory publication barrier; b1/b2/b4/b8 add 8/4/2/1 barriers. This measures additional rendezvous around fixed work; immutable reads may be hoisted.
- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.
