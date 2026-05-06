# Advanced Investigation Plans Index (Lecture Notes Track)

This index points to detailed lecture-note style plans for post-25 advanced investigations.
Canonical plan files now live next to their experiment reports under `experiments/<id>/plan.md`.

## Recommended Execution Phases
1. Phase A: core primitives (radix sort, subgroup ops, occupancy modeling).
2. Phase B: rendering data systems (tiled assignment, culling pipelines, GPU-driven blocks).
3. Phase C: architecture depth (BVH layout, ray-friendly layouts, temporal coherence).
4. Phase D: systems and platform depth (persistent queues, async overlap, cross-GPU validation).

## Advanced Investigation Plans
- [Advanced Investigation 01: Radix Sort on GPU](../experiments/34_radix_sort_gpu/plan.md): Multi-pass radix sorting for key-only and key-value data.
- [Advanced Investigation 02: BVH Node Layout Experiments](../experiments/36_bvh_node_layout/plan.md): Node representation impact on traversal efficiency.
- [Advanced Investigation 03: Frustum Culling vs Clustered Culling](../experiments/38_frustum_vs_clustered_culling/plan.md): Comparative rendering-style visibility pipelines.
- [Advanced Investigation 04: Tiled Light Assignment](../experiments/39_tiled_light_assignment/plan.md): Light-to-tile list construction strategies.
- [Advanced Investigation 05: Persistent Threads and Work Queues](../experiments/40_persistent_threads_work_queues/plan.md): Dynamic scheduling for irregular workloads.
- [Advanced Investigation 06: Subgroup Operations Study](../experiments/41_subgroup_operations_study/plan.md): Warp/wave-level intrinsics versus shared-memory patterns.
- [Advanced Investigation 07: Async Compute Overlap](../experiments/42_async_compute_overlap/plan.md): Overlapping compute, transfer, and independent stages.
- [Advanced Investigation 08: Occupancy Modeling Against Vendor Guidance](../experiments/35_occupancy_modeling/plan.md): Resource-pressure interpretation of measured trends.
- [Advanced Investigation 09: Memory System Study for Ray-Friendly Layouts](../experiments/43_ray_friendly_memory_layouts/plan.md): Traversal-friendly layout behavior under coherent/incoherent query patterns.
- [Advanced Investigation 10: Frame-to-Frame Coherence Studies](../experiments/37_frame_to_frame_coherence/plan.md): Temporal stability and ordering reuse effects.
- [Advanced Investigation 11: GPU-Driven Pipeline Building Blocks](../experiments/44_gpu_driven_pipeline_blocks/plan.md): Compose culling, compaction, bucketing, and argument-like generation.
- [Advanced Investigation 12: Reproducibility and Cross-GPU Comparison](../experiments/45_cross_gpu_reproducibility/plan.md): Trend stability across architectures and vendors.
