# GPU Memory Access Optimization: Coalescing, Layout, and Cache Behavior

Public project title: `GPU Memory Access Optimization: Coalescing, Layout, and Cache Behavior`  
Repository and current runtime target: `gpu-memory-layout-playground` / `gpu_memory_layout_experiments`

This repository is a Vulkan compute benchmarking project focused on one question: how much GPU performance is won or lost by the way memory is laid out and accessed?

The project is positioned as results-backed engineering work, not a toy benchmark collection. It studies layout, coalescing, locality, bandwidth saturation, and shared-memory staging with reproducible Vulkan compute experiments and artifact-driven reports.

## Problem
- GPUs are frequently bandwidth-bound rather than ALU-bound.
- Memory access patterns often dominate kernel performance.
- Small indexing or layout changes can collapse effective throughput even when arithmetic stays constant.
- Real GPU engineering work depends on understanding coalescing, cache behavior, and when on-chip staging is actually worth the cost.

## Results Snapshot
| Experiment | Improvement | Current evidence |
| --- | --- | --- |
| SoA vs AoS | `+2742%` GPU-time speedup (`28.42x` faster) and `+2032%` effective-bandwidth gain for `SoA` | Experiment 06, `64 MiB`, `1,000,000` elements |
| Coalesced vs Strided | `+1298%` to `+8808%` GPU-time speedup for coalesced access depending on stride | Experiment 11, stride sweep `1, 2, 4, 8, 16, 32, 64`, `128 MiB` |
| Cache line utilization / reuse | Bounded reuse is about `1.32x` faster than full-span replay | Experiment 14, `32 MiB` locality study |
| Memory bandwidth saturation | Read-only traffic sustains about `899 GB/s` from `96-512 MiB`; read-write copy sustains about `419 GB/s` | Experiment 15 saturation sweep |
| Shared memory vs global memory | Current tiled kernel is `0.65%-2.85%` slower than direct global access | Experiment 16, reuse-radius sweep |

## Methodology
| Dimension | Current setup |
| --- | --- |
| GPU | Desktop: `NVIDIA GeForce RTX 2080 SUPER` (`Vulkan 1.4.325`, driver `2480242688`) |
| Mobile track | `Adreno` validation is planned as a cross-GPU extension, not yet part of the measured baseline |
| Measurement | GPU timestamp queries, median GPU time as the primary metric, `p95` for stability |
| Data sizes | `32 MiB`, `64 MiB`, `128 MiB`, and `512 MiB` depending on the experiment |
| Outputs | Raw JSON exports, CSV summaries, PNG charts, and per-experiment `results.md` reports |

## Core Experiment Set
| Status | Experiment | Purpose |
| --- | --- | --- |
| Implemented | AoS vs SoA | Layout efficiency for field-wise kernels |
| Implemented | Coalesced vs Strided | Memory transaction efficiency under stride |
| Implemented | Cache line utilization and reuse distance | Locality and replay cost |
| Implemented | Memory bandwidth saturation | Steady-state throughput limits |
| Implemented | Shared memory vs global memory | Whether staging overhead is repaid |
| Implemented | Warp-level coalescing alignment | Aligned vs misaligned contiguous accesses |
| Implemented | Cache thrashing | Random vs sequential working sets |
| Priority next | Cross-GPU validation | Desktop vs mobile/Adreno behavior |

## Key Findings
- Coalesced access is the dominant good-path baseline. The first loss of coalescing causes the largest collapse in effective throughput.
- `SoA` is the correct default layout for field-wise access on the current workload. `AoS` wastes bandwidth badly.
- Cache-friendly bounded reuse materially outperforms full-span replay, even without hardware counters.
- Shared memory is not automatically faster. The current staging kernel does more work without repaying that overhead.
- Size sweeps matter. Small transfers do not represent the sustained bandwidth region.

## Visuals
Current graphs:
- [AoS vs SoA GB/s chart](./experiments/06_aos_vs_soa/results/charts/aos_vs_soa_gbps_vs_size.png)
- [Coalesced vs Strided slowdown chart](./experiments/11_coalesced_vs_strided/results/charts/coalesced_vs_strided_slowdown_vs_stride_1.png)
- [Bandwidth saturation GB/s chart](./experiments/15_bandwidth_saturation_sweep/results/charts/bandwidth_saturation_gbps_vs_size.png)
- [Shared memory vs direct-global speedup chart](./experiments/16_shared_memory_tiling/results/charts/shared_memory_tiling_speedup_vs_direct.png)

![AoS vs SoA GB/s](./experiments/06_aos_vs_soa/results/charts/aos_vs_soa_gbps_vs_size.png)

![Coalesced vs Strided Slowdown](./experiments/11_coalesced_vs_strided/results/charts/coalesced_vs_strided_slowdown_vs_stride_1.png)

![Bandwidth Saturation](./experiments/15_bandwidth_saturation_sweep/results/charts/bandwidth_saturation_gbps_vs_size.png)

Profiler screenshots to add:
- Warp-level alignment capture: aligned vs misaligned coalescing on the same warp-sized load.
- Cache-thrashing capture: sequential vs random access with memory-stall or cache-hit counters.
- Shared-memory staging capture: `shared_tiled` vs `direct_global` stall breakdown.

## Engineering Insight
### Why coalescing matters
Warps and waves issue many lane requests together. When neighboring lanes read neighboring addresses, the memory system can satisfy the group with fewer transactions. When access becomes strided or misaligned, the hardware moves more bytes for the same useful work.

### How GPU memory transactions work
The GPU does not service each lane as an isolated scalar load. Lane requests are merged into cache-line or transaction-sized memory operations. Effective bandwidth falls when the transaction footprint grows faster than the useful-data footprint.

### Relation to SIMD and warps
Poor coalescing is the memory-side equivalent of wasted SIMD efficiency. Branch divergence wastes active lanes; bad memory layout wastes transferred bytes. Both reduce how much useful work each issued warp or wave actually produces.

## Documentation
- [Research Overview](./docs/research_overview.md)
- [Development Principles and Plan](./docs/development_principles_and_plan.md)
- [Core Experiment Plans Index](./docs/core_experiment_plans_index.md)
- [Advanced Investigations Roadmap](./docs/advanced_investigations_roadmap.md)
