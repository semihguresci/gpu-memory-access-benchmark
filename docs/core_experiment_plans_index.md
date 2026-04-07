<!-- Generated from config/experiment_manifest.json via scripts/generate_experiment_plan_index.py -->
# Core Experiment Plans Index (Lecture Notes Track)

This index points to the detailed lecture-note style plan for each core experiment.

## Recommended Reading Order
- Start with Experiments 01-05 to stabilize benchmark methodology and execution-model intuition.
- Continue with Experiments 06-10 for layout and alignment design rules.
- Use Experiments 11-15 to map access pattern, cache behavior, and saturation.
- Use Experiments 16-20 to build architecture-aware optimization intuition.
- Finish with Experiments 21-25 to assemble practical parallel primitives and capstone systems.

## Memory Optimization Sequence
- [Experiment 06: AoS vs SoA](experiment_plans/06_aos_vs_soa.md)
- [Experiment 11: Coalesced vs Strided Access](experiment_plans/11_coalesced_vs_strided.md)
- [Experiment 14: Read Reuse and Cache Locality](experiment_plans/14_read_reuse_cache_locality.md)
- [Experiment 15: Bandwidth Saturation Sweep](experiment_plans/15_bandwidth_saturation_sweep.md)
- [Experiment 16: Shared or Workgroup Memory Tiling](experiment_plans/16_shared_memory_tiling.md)

## Core Experiment Plans
- [Experiment 01: Dispatch Basics](experiment_plans/01_dispatch_basics.md): Minimal Vulkan compute dispatch, correctness path, and baseline GPU timing.
- [Experiment 02: Local Size Sweep](experiment_plans/02_local_size_sweep.md): Workgroup sizing and execution efficiency tradeoffs.
- [Experiment 03: Memory Copy Baseline](experiment_plans/03_memory_copy_baseline.md): Raw buffer read/write/copy throughput characterization.
- [Experiment 04: Sequential Indexing](experiment_plans/04_sequential_indexing.md): Ideal contiguous thread-to-data mapping as a good-path baseline.
- [Experiment 05: Global ID Mapping Variants](experiment_plans/05_global_id_mapping_variants.md): Direct, offset, and grid-stride mapping behavior.
- [Experiment 06: AoS vs SoA](experiment_plans/06_aos_vs_soa.md): Array-of-Structures versus Structure-of-Arrays layout efficiency.
- [Experiment 07: AoSoA or Blocked Layout](experiment_plans/07_aosoa_blocked_layout.md): Hybrid layout balancing vector locality and contiguous field access.
- [Experiment 08: std430 vs std140 vs Packed](experiment_plans/08_std430_std140_packed.md): Shader buffer layout standards and padding cost.
- [Experiment 09: vec3, vec4, and Padding Costs](experiment_plans/09_vec3_vec4_padding_costs.md): Impact of vector shape choice on storage efficiency and bandwidth.
- [Experiment 10: Scalar Type Width Sweep](experiment_plans/10_scalar_type_width_sweep.md): Precision-width tradeoffs: 32-bit, 16-bit, and narrower storage.
- [Experiment 11: Coalesced vs Strided Access](experiment_plans/11_coalesced_vs_strided.md): Contiguous and strided load behavior.
- [Experiment 12: Gather Access Pattern](experiment_plans/12_gather_access_pattern.md): Indirect indexed reads through an index buffer.
- [Experiment 13: Scatter Access Pattern](experiment_plans/13_scatter_access_pattern.md): Indirect indexed writes and contention behavior.
- [Experiment 14: Read Reuse and Cache Locality](experiment_plans/14_read_reuse_cache_locality.md): Temporal locality and reuse-distance effects.
- [Experiment 15: Bandwidth Saturation Sweep](experiment_plans/15_bandwidth_saturation_sweep.md): Scaling data volume until practical bandwidth plateau.
- [Experiment 16: Shared or Workgroup Memory Tiling](experiment_plans/16_shared_memory_tiling.md): Staging data in on-chip memory for reuse.
- [Experiment 17: Tile Size Sweep](experiment_plans/17_tile_size_sweep.md): Tradeoff between reuse, shared-memory pressure, and occupancy.
- [Experiment 18: Register Pressure Proxy Study](experiment_plans/18_register_pressure_proxy.md): Effect of increased per-thread temporary state.
- [Experiment 19: Branch Divergence](experiment_plans/19_branch_divergence.md): Control-flow divergence within warp or wave execution.
- [Experiment 20: Barrier and Synchronization Cost](experiment_plans/20_barrier_synchronization_cost.md): Synchronization overhead characterization.
- [Experiment 21: Parallel Reduction](experiment_plans/21_parallel_reduction.md): Reduction patterns from naive to tree and shared-memory optimized.
- [Experiment 22: Prefix Sum or Scan](experiment_plans/22_prefix_sum_scan.md): Inclusive/exclusive scan as a foundational parallel primitive.
- [Experiment 23: Histogram and Atomic Contention](experiment_plans/23_histogram_atomic_contention.md): Atomic update contention and privatization strategies.
- [Experiment 24: Stream Compaction](experiment_plans/24_stream_compaction.md): Flag, scan, and compact-write pipeline.
- [Experiment 25: Spatial Binning or Clustered Culling Capstone](experiment_plans/25_spatial_binning_clustered_culling_capstone.md): Rendering-style compute pipeline combining prior primitives.

## Priority Extensions Beyond Core 25
- [Experiment 26: Warp-Level Coalescing Alignment](experiment_plans/26_warp_level_coalescing_alignment.md): Aligned vs misaligned contiguous accesses at warp granularity.
- [Experiment 27: Cache Thrashing, Random vs Sequential](experiment_plans/27_cache_thrashing_random_vs_sequential.md): Healthy locality versus deliberate cache defeat.
- [Experiment 28: Device-Local vs Host-Visible Heap Placement](experiment_plans/28_device_local_vs_host_visible_heap_placement.md): Dispatch-only and end-to-end cost of host-visible buffers versus staged device-local placement.
- [Experiment 29: Shared Memory Bank Conflict Study](experiment_plans/29_shared_memory_bank_conflict_study.md): Stride-driven shared-memory bank conflicts and the padding fix.
- [Experiment 30: Subgroup Reduction Variants](experiment_plans/30_subgroup_reduction_variants.md): Compare shared-tree reduction with subgroup-assisted reduction.
- [Experiment 31: Subgroup Scan Variants](experiment_plans/31_subgroup_scan_variants.md): Block-local inclusive scan using shared memory versus subgroup intrinsics.
- [Experiment 32: Subgroup Stream Compaction Variants](experiment_plans/32_subgroup_stream_compaction_variants.md): Per-workgroup compaction using shared atomics versus subgroup ballot ranking.
- [Experiment 33: 2D Locality and Transpose Study](experiment_plans/33_two_dimensional_locality_transpose_study.md): Row-major copy versus naive and tiled transpose access patterns.
