# GPU Memory Access Optimization: Coalescing, Layout, and Cache Behavior

Public project title: `GPU Memory Access Optimization: Coalescing, Layout, and Cache Behavior`  
Repository and current runtime target: `gpu-memory-layout-playground` / `gpu_memory_layout_experiments`

This repository is a Vulkan compute benchmarking project focused on one question: how much GPU performance is won or lost by the way memory is laid out and accessed?

The project is positioned as results-backed engineering work. It studies layout, coalescing, locality, bandwidth saturation, and shared-memory staging with reproducible Vulkan compute experiments and artifact-driven reports.

## Current Scope
The repository currently tracks `48` enabled Vulkan compute experiments:
- Experiments `01-25`: core benchmark foundations, layout studies, access patterns, on-chip memory, execution-model probes, and parallel primitives.
- Experiments `26-33`: priority memory-system extensions, including warp alignment, cache thrashing, heap placement, bank conflicts, subgroup variants, and 2D locality.
- Experiments `34-48`: advanced rendering and systems investigations, including radix sort, BVH layout, culling, tiled light assignment, persistent queues, subgroup operations, async overlap, ray-friendly layouts, GPU-driven pipeline blocks, cross-GPU reproducibility, dependent-load latency, access-pattern factorials, and buffer-versus-image representations.

Canonical experiment metadata lives in [`config/experiment_manifest.json`](./config/experiment_manifest.json). Each experiment owns its local notes and artifacts under `experiments/<id>/`, including `README.md`, `plan.md`, `results.md`, generated `results/` charts/tables, and archived `runs/` when data has been collected.

The repository tracks source, scripts, and Markdown reports. Generated `results/`
and archived `runs/` directories are ignored by Git; collect data and regenerate
artifacts locally with the commands below.

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
python scripts/run_experiment_data_collection.py --experiment 06_aos_vs_soa --binary build-tests-vs/Release/gpu_memory_layout_experiments.exe --iterations 5 --warmup 2
```

Regenerate derived tables and charts for an experiment:

```powershell
python scripts/generate_experiment_artifacts.py --experiment 06_aos_vs_soa
python scripts/refresh_experiment_reports.py --experiment 06_aos_vs_soa
```

Artifact generation can also use existing run archives. Experiments 46–48 rebuild
both tables and charts when the current `benchmark_results.json` is absent.

Useful flags:
- `--size 1G` overrides the manifest default scratch size.
- `--size-preset {tiny,small,medium,large,xlarge}` expands to 4M/32M/128M/256M/1G (mutually exclusive with `--size`).
- `--device 0` selects a physical device by enumeration index; `--device geforce` matches a deviceName substring (case-insensitive; empty selects the first compute-capable device).
- `--runs 3` performs independent benchmark invocations and archives each run with a `run<index>` label suffix.
- `--validation` enables Vulkan validation layers for the benchmark run.
- `--verbose-progress` enables detailed process logs; default runs stay quiet.
- `--binary <path>` pins the benchmark executable when a stale build path would otherwise be selected.

## Results Snapshot
The current reports use measurement revision `2026-11-08-remaining-review-fixes`,
with experiments 38 and 40 refreshed under `2026-11-08-contention-merge` after
integrating the workgroup contention optimizations from `main`.
Together, the current exports pass **6,105 of 6,105 timed rows across all 48 experiments**.
All 48 reports were refreshed, with **192 derived tables and 191 charts** verified
against their recorded source and tool hashes. Reports identify the exact raw
export, matching archive, binary, source-tree, and shader-bundle provenance.

Older archived measurements use earlier workload, synchronization, validation, or
byte-accounting definitions and remain separate from the current revision. See
the [review fixes and validation record](./docs/experiment_integrity_fixes.md).

| Experiment | Measured comparison | Current evidence |
| --- | --- | --- |
| SoA vs AoS | Matching eight field reads and four field stores with a common 48-byte payload | [Experiment 06 results](./experiments/06_aos_vs_soa/results.md) |
| Coalesced vs Strided | Strides `1, 2, 4, 8, 16, 32, 64`; physical span also changes | [Experiment 11 results](./experiments/11_coalesced_vs_strided/results.md) |
| Read reuse | Invocation-index schedules over repeated source values; temporal execution order is not guaranteed | [Experiment 14 results](./experiments/14_read_reuse_cache_locality/results.md) |
| Memory bandwidth sweep | Read/write/copy requested-byte throughput over the configured size range | [Experiment 15 results](./experiments/15_bandwidth_saturation_sweep/results.md) |
| Shared memory | Direct versus tiled stencil at matching reuse radius | [Experiment 16 results](./experiments/16_shared_memory_tiling/results.md) |

## Methodology
| Dimension | Current setup |
| --- | --- |
| GPU | Desktop: `NVIDIA GeForce RTX 2080 SUPER` (`Vulkan 1.4.341`, NVIDIA driver `610.88`) |
| Mobile track | `Adreno` validation is planned as a cross-GPU extension, not yet part of the measured baseline |
| Measurement | GPU timestamp queries, median GPU time as the primary metric, `p95` for stability |
| Current refresh | `32 MiB` total scratch per experiment, except experiments 34, 47, and 48 at `128 MiB`; five timed iterations and two warmups |
| Outputs | Raw JSON exports, CSV summaries, PNG charts, and per-experiment `results.md` reports |

## Validation
As of the 2026-11-08 review, the Release build passes **57 CTest cases** (50 unit
and seven integration cases) and **74 Python tests**. One integration case runs
a **26-experiment GPU smoke matrix**; the full 48-experiment collection above is
separate run evidence. The modified production translation units pass the required
GCC warning checks, and changed code passes the repository clang-tidy checks.

After configuring the test preset, build and run the automated suites:

```powershell
cmake --build --preset tests-vs-release --target gpu_memory_layout_tests
ctest --test-dir build-tests-vs -C Release --output-on-failure
python -m unittest discover -s scripts/tests -v
python -m unittest discover -s tests/scripts -v
```

GPU-dependent tests require a usable Vulkan device and timestamp support. The
[test coverage map](./docs/tests/test_coverage.md) describes skip conditions,
regressions, and remaining coverage limits. The
[review record](./docs/experiment_integrity_fixes.md#remaining-review-fixes)
documents descriptor-limit guards, fresh-output validation, synchronization,
traffic accounting, and archive-only artifact checks.

## Experiment Catalog
| Track | Experiments | Focus |
| --- | --- | --- |
| Core | `01-25` | Benchmark foundations, memory layout, access patterns, cache behavior, shared memory, synchronization, and parallel primitives |
| Extension | `26-33` | Warp alignment, cache thrashing, heap placement, bank conflicts, subgroup primitive variants, and 2D locality |
| Advanced | `34-48` | Sorting, rendering data structures, culling, tiled assignment, persistent queues, async overlap, GPU-driven blocks, dependent-load latency, access-pattern factorials, buffer-versus-image representations, and cross-GPU comparison |

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
| Enabled | BVH Node Layout | Compact versus padded storage in a flat node-access proxy |
| Enabled | GPU-Driven Pipeline Blocks | Staged versus fused compute pipeline building blocks |
| Enabled | Cross-GPU Reproducibility | Deterministic probes for cross-run and cross-GPU comparison |
| Enabled | Dependent Load Latency | Pointer-chase chains separating latency from memory-level parallelism |
| Enabled | Block Reorder Factorial | 2x2 factorial over block order and within-block lane order with residency controls |
| Enabled | Buffer Vs Image Representation | Same 2D dataset across flat buffer layouts and storage/sampled image paths |

Full plan indexes:
- [Core Experiment Plans Index](./docs/core_experiment_plans_index.md)
- [Advanced Investigation Plans Index](./docs/advanced_investigation_plans_index.md)

## Reading the Results
- Layout comparisons keep active field operations constant; physical transaction counts still require hardware evidence.
- GPU time, logical elements, and estimated/requested bytes are separate quantities; each report documents its scope.
- Experiment 22 GB/s estimates source-level storage-buffer requests, including per-invocation carry loads; it does not measure physical DRAM traffic.
- Device comparisons retain validation mode, warmup count, and timed-iteration count so different run settings are not pooled.
- Shared-memory and register studies keep useful work constant. Source declarations alone do not establish occupancy or spills.
- Actual allocation flags and device limits matter; host-visible memory does not necessarily identify a distinct heap.
- The current small-size, single-GPU refresh is exploratory. Larger sweeps and additional devices are needed for sustained-bandwidth and portability conclusions.

## Visuals
Charts are generated locally in `experiments/<id>/results/charts/`. They are
ignored by Git, so a fresh clone needs data collection and artifact generation
before those PNGs are available. Example outputs:

| Experiment and report | Generated chart filename |
| --- | --- |
| [AoS vs SoA](./experiments/06_aos_vs_soa/results.md) | `aos_vs_soa_gbps_vs_size.png` |
| [Coalesced vs Strided](./experiments/11_coalesced_vs_strided/results.md) | `coalesced_vs_strided_slowdown_vs_stride_1.png` |
| [Bandwidth saturation](./experiments/15_bandwidth_saturation_sweep/results.md) | `bandwidth_saturation_gbps_vs_size.png` |
| [Shared memory vs direct-global](./experiments/16_shared_memory_tiling/results.md) | `shared_memory_tiling_speedup_vs_direct.png` |

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
Its **Artifacts Only** action works from existing logs without requiring a benchmark
binary or valid benchmark iteration settings.

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
- [Review Fixes and Validation Record](./docs/experiment_integrity_fixes.md)
- [Test Coverage](./docs/tests/test_coverage.md)
