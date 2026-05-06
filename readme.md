# GPU Memory Access Optimization: Coalescing, Layout, and Cache Behavior

Public project title: `GPU Memory Access Optimization: Coalescing, Layout, and Cache Behavior`  
Repository and current runtime target: `gpu-memory-layout-playground` / `gpu_memory_layout_experiments`

This repository is a Vulkan compute benchmarking project focused on one question: how much GPU performance is won or lost by the way memory is laid out and accessed?

The project is positioned as results-backed engineering work. It studies layout, coalescing, locality, bandwidth saturation, and shared-memory staging with reproducible Vulkan compute experiments and artifact-driven reports.

## Current Scope
The repository currently tracks `45` enabled Vulkan compute experiments:
- Experiments `01-25`: core benchmark foundations, layout studies, access patterns, on-chip memory, execution-model probes, and parallel primitives.
- Experiments `26-33`: priority memory-system extensions, including warp alignment, cache thrashing, heap placement, bank conflicts, subgroup variants, and 2D locality.
- Experiments `34-45`: advanced rendering and systems investigations, including radix sort, BVH layout, culling, tiled light assignment, persistent queues, subgroup operations, async overlap, ray-friendly layouts, GPU-driven pipeline blocks, and cross-GPU reproducibility.

Canonical experiment metadata lives in [`config/experiment_manifest.json`](./config/experiment_manifest.json). Each experiment owns its local notes and artifacts under `experiments/<id>/`, including `README.md`, `plan.md`, `results.md`, generated `results/` charts/tables, and archived `runs/` when data has been collected.

## Problem
- GPUs are frequently bandwidth-bound rather than ALU-bound.
- Memory access patterns often dominate kernel performance.
- Small indexing or layout changes can collapse effective throughput even when arithmetic stays constant.
- Real GPU engineering work depends on understanding coalescing, cache behavior, and when on-chip staging is actually worth the cost.

## Quick Start
Configure and build the Visual Studio test preset with shader auto-compilation enabled:

```powershell
cmake --preset windows-tests-vs
cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments
```

Run one experiment and collect the raw benchmark JSON into its experiment-local run archive:

```powershell
python scripts/run_experiment_data_collection.py --experiment 06_aos_vs_soa --iterations 5 --warmup 2
```

Regenerate derived tables and charts for an experiment:

```powershell
python scripts/generate_experiment_artifacts.py --experiment 06_aos_vs_soa
```

Useful flags:
- `--size 1G` overrides the manifest default scratch size.
- `--validation` enables Vulkan validation layers for the benchmark run.
- `--verbose-progress` enables detailed process logs; default runs stay quiet.
- `--binary <path>` pins the benchmark executable when a stale build path would otherwise be selected.

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

## Experiment Catalog
| Track | Experiments | Focus |
| --- | --- | --- |
| Core | `01-25` | Benchmark foundations, memory layout, access patterns, cache behavior, shared memory, synchronization, and parallel primitives |
| Extension | `26-33` | Warp alignment, cache thrashing, heap placement, bank conflicts, subgroup primitive variants, and 2D locality |
| Advanced | `34-45` | Sorting, rendering data structures, culling, tiled assignment, persistent queues, async overlap, GPU-driven blocks, and cross-GPU comparison |

Representative enabled studies:

| Status | Experiment | Purpose |
| --- | --- | --- |
| Enabled | AoS vs SoA | Layout efficiency for field-wise kernels |
| Enabled | Coalesced vs Strided | Memory transaction efficiency under stride |
| Enabled | Cache line utilization and reuse distance | Locality and replay cost |
| Enabled | Memory bandwidth saturation | Steady-state throughput limits |
| Enabled | Shared memory vs global memory | Whether staging overhead is repaid |
| Enabled | Warp-level coalescing alignment | Aligned vs misaligned contiguous accesses |
| Enabled | Cache thrashing | Random vs sequential working sets |
| Enabled | Radix Sort on GPU | Multi-pass key sorting and digit-width tradeoffs |
| Enabled | BVH Node Layout | Compact versus padded node storage and traversal locality |
| Enabled | GPU-Driven Pipeline Blocks | Staged versus fused compute pipeline building blocks |
| Enabled | Cross-GPU Reproducibility | Deterministic probes for cross-run and cross-GPU comparison |

Full plan indexes:
- [Core Experiment Plans Index](./docs/core_experiment_plans_index.md)
- [Advanced Investigation Plans Index](./docs/advanced_investigation_plans_index.md)

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

## GUI Runner
For local experiment management, a Tkinter runner is available at `scripts/experiment_gui.py`.

Launch it from the repository root:

```powershell
python scripts/experiment_gui.py
```

What it wraps:
- build via the repo CMake presets
- benchmark execution through `scripts/run_experiment_data_collection.py`
- artifact regeneration through `scripts/generate_experiment_artifacts.py`

The GUI reads `config/experiment_manifest.json`, lets you multi-select experiments, streams live logs, and can stop the active process tree on Windows.

## Engineering Insight
### Why coalescing matters
Warps and waves issue many lane requests together. When neighboring lanes read neighboring addresses, the memory system can satisfy the group with fewer transactions. When access becomes strided or misaligned, the hardware moves more bytes for the same useful work.

### How GPU memory transactions work
The GPU does not service each lane as an isolated scalar load. Lane requests are merged into cache-line or transaction-sized memory operations. Effective bandwidth falls when the transaction footprint grows faster than the useful-data footprint.

### Relation to SIMD and warps
Poor coalescing is the memory-side equivalent of wasted SIMD efficiency. Branch divergence wastes active lanes; bad memory layout wastes transferred bytes. Both reduce how much useful work each issued warp or wave actually produces.

## Documentation
- [Documentation Index](./docs/README.md)
- [Research Overview](./docs/research_overview.md)
- [Project Architecture](./docs/architecture.md)
- [Development Principles and Plan](./docs/development_principles_and_plan.md)
- [Core Experiment Plans Index](./docs/core_experiment_plans_index.md)
- [Advanced Investigation Plans Index](./docs/advanced_investigation_plans_index.md)
- [Advanced Investigations Roadmap](./docs/advanced_investigations_roadmap.md)
