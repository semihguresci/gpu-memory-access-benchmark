# Research Overview

## Title
GPU Memory Access Optimization: Coalescing, Layout, and Cache Behavior

## Thesis
This repository studies the most important practical GPU-memory question in rendering-adjacent compute work: how much performance depends on matching data layout and access order to the way warps or waves fetch memory.

The project is intentionally results-first. Every major claim should be backed by measured GPU time, machine-readable outputs, generated charts, and a short written interpretation with clear limitations.

## Problem Statement
- GPUs are often bandwidth-bound before they are arithmetic-bound.
- Memory access patterns dominate throughput on many real kernels.
- Layout, stride, locality, and staging decisions can create order-of-magnitude differences in performance.
- The goal is to turn those facts into reproducible engineering evidence instead of generic advice.

## Current Experiment Program
| Category | Experiment | Status | Why it matters |
| --- | --- | --- | --- |
| Data layout | AoS vs SoA | Implemented | Tests whether field-wise access should use contiguous arrays |
| Transaction efficiency | Coalesced vs Strided | Implemented | Directly measures what happens when contiguous lane access is lost |
| Cache behavior | Read reuse and locality | Implemented | Measures bounded reuse versus full-span replay |
| Saturation | Memory bandwidth sweep | Implemented | Shows when the GPU reaches sustained throughput instead of overhead-dominated behavior |
| On-chip staging | Shared memory vs global memory | Implemented | Tests whether workgroup memory repays its setup and barrier cost |
| Warp behavior | Aligned vs misaligned coalescing | Implemented | Isolates transaction-boundary penalties within warp-sized access windows |
| Cache defeat | Random vs sequential working sets | Implemented | Separates healthy locality from deliberate cache thrashing |
| Heap placement | Device-local vs host-visible | Implemented | Separates dispatch-only bandwidth from staging overhead on discrete GPUs |
| Shared-memory pathologies | Bank conflicts and padding fix | Implemented | Shows when on-chip memory loses because the access pattern is conflict-heavy |
| Subgroup primitives | Reduction, scan, and compaction variants | Implemented | Extends the primitive track with wave- or warp-level fast paths |
| 2D locality | Matrix transpose study | Implemented | Connects the memory story to image-style and rendering-adjacent workloads |
| Portability | Desktop vs Adreno/mobile | Planned priority extension | Distinguishes architecture-specific behavior from robust trends |

## Current Methodology
| Dimension | Baseline |
| --- | --- |
| Desktop GPU | `NVIDIA GeForce RTX 2080 SUPER` |
| API | `Vulkan 1.4.325` |
| Timing method | GPU timestamp queries |
| Main metrics | median GPU time, `p95`, effective throughput, effective GB/s |
| Example data sizes | `32 MiB`, `64 MiB`, `128 MiB`, `512 MiB` |
| Artifact policy | raw JSON + CSV tables + PNG charts + `results.md` interpretation |

Adreno or other mobile validation is not yet part of the measured baseline. It is explicitly planned as a follow-up so the project can separate desktop-specific effects from broader GPU-memory rules.

## Current Results
| Experiment | Headline result | Supporting artifact |
| --- | --- | --- |
| Experiment 06: AoS vs SoA | `SoA` is `28.42x` faster in GPU time and `21.32x` higher in effective bandwidth than `AoS` on the current dataset | [AoS vs SoA results](../experiments/06_aos_vs_soa/results.md) |
| Experiment 11: Coalesced vs Strided | Coalesced `stride_1` is `89.08x` faster than `stride_2` and still `13.98x` faster than the best measured strided case, `stride_16` | [Coalesced vs Strided results](../experiments/11_coalesced_vs_strided/results.md) |
| Experiment 14: Read Reuse and Cache Locality | Bounded reuse stays about `1.32x` faster than full-span replay | [Reuse and locality results](../experiments/14_read_reuse_cache_locality/results.md) |
| Experiment 15: Bandwidth Saturation Sweep | Read-only traffic sustains about `899 GB/s` from `96 MiB` onward; read-write copy sustains about `419 GB/s` | [Bandwidth saturation results](../experiments/15_bandwidth_saturation_sweep/results.md) |
| Experiment 16: Shared Memory Tiling | The current shared-memory kernel is slightly slower than direct global access across all measured reuse radii | [Shared memory tiling results](../experiments/16_shared_memory_tiling/results.md) |

## Key Findings
- Coalescing is the single strongest good-path rule in the current dataset.
- Layout decisions matter at an order-of-magnitude scale, not as a minor optimization.
- Locality matters, but timing-only cache studies need careful scoping and explicit caveats.
- Shared memory must be justified with measured reuse and occupancy tradeoffs. It is not a default win.
- Saturation studies need size sweeps. Single-size microbenchmarks are too easy to misread.

## Engineering Insight
### Why coalescing matters
GPUs issue work in groups of lanes. If those lanes touch contiguous addresses, the memory subsystem can merge requests into a small number of transactions. If the lane addresses spread out, the same useful work consumes more memory transactions and effective throughput collapses.

### How GPU memory transactions work
Useful bytes are only part of the story. The memory system moves transaction-sized chunks, not just the individual scalar values a kernel requested. That means layout and alignment determine whether a warp moves mostly useful bytes or mostly wasted bytes.

### Relation to SIMD, warps, and waves
This is the memory equivalent of SIMD efficiency. Branch divergence wastes execution lanes. Poor coalescing wastes memory bandwidth. Both reduce the useful work done per issued instruction group.

## Visual Package
Current charts:
- [AoS vs SoA GB/s](../experiments/06_aos_vs_soa/results/charts/aos_vs_soa_gbps_vs_size.png)
- [Coalesced vs Strided slowdown](../experiments/11_coalesced_vs_strided/results/charts/coalesced_vs_strided_slowdown_vs_stride_1.png)
- [Reuse locality speedup](../experiments/14_read_reuse_cache_locality/results/charts/read_reuse_cache_locality_speedup_vs_full_span.png)
- [Bandwidth saturation GB/s](../experiments/15_bandwidth_saturation_sweep/results/charts/bandwidth_saturation_gbps_vs_size.png)
- [Shared-memory speedup vs direct](../experiments/16_shared_memory_tiling/results/charts/shared_memory_tiling_speedup_vs_direct.png)

Required profiler screenshots for the next reporting pass:
- aligned vs misaligned coalescing capture
- sequential vs random cache-behavior capture
- shared-memory vs direct-global stall or memory-transaction capture

## Priority Extensions
The current repository now includes the extension studies that complete the next layer of the memory-access narrative:

1. [Experiment 26: Warp-Level Coalescing Alignment](experiment_plans/26_warp_level_coalescing_alignment.md)
2. [Experiment 27: Cache Thrashing, Random vs Sequential](experiment_plans/27_cache_thrashing_random_vs_sequential.md)
3. [Experiment 28: Device-Local vs Host-Visible Heap Placement](experiment_plans/28_device_local_vs_host_visible_heap_placement.md)
4. [Experiment 29: Shared Memory Bank Conflict Study](experiment_plans/29_shared_memory_bank_conflict_study.md)
5. [Experiment 30: Subgroup Reduction Variants](experiment_plans/30_subgroup_reduction_variants.md)
6. [Experiment 31: Subgroup Scan Variants](experiment_plans/31_subgroup_scan_variants.md)
7. [Experiment 32: Subgroup Stream Compaction Variants](experiment_plans/32_subgroup_stream_compaction_variants.md)
8. [Experiment 33: 2D Locality and Transpose Study](experiment_plans/33_two_dimensional_locality_transpose_study.md)

These additions make the project materially stronger because they turn the current memory story into a cleaner engineering sequence:
- layout choice
- coalescing quality
- cache locality and cache defeat
- buffer heap placement
- bandwidth saturation
- on-chip staging tradeoffs
- bank-conflict sensitivity
- subgroup-level primitive design
- 2D transpose locality
- cross-GPU validation

## Positioning
The strongest public framing for this project is:

`I build Vulkan compute benchmarks that explain how GPU memory access really behaves, and I can connect layout and access decisions to measured performance outcomes.`
