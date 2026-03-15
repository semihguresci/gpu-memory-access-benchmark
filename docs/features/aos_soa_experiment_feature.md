# AoS vs SoA Experiment Feature (06)

## Source
- `src/experiments/aos_soa_experiment.cpp`

## Purpose
Measures compute performance differences between Array-of-Structs (AoS) and Struct-of-Arrays (SoA) memory layouts.

## Data Model
- Particle payload uses 8 floats per logical element.
- Host-side AoS struct is compile-time validated with layout assertions.

## Pipeline Model
- AoS path:
  - one storage buffer
  - one descriptor binding
- SoA path:
  - eight storage buffers
  - eight descriptor bindings
- Both paths use push constants for particle count and a fixed workgroup size (`256`).

## Size Sweep
- Preferred counts: `1M`, `5M`, `10M` particles when possible.
- Falls back to smaller powers when scratch budget is limited.
- Final fallback uses maximum feasible particle count if non-zero.

## Measurement Path
- Uses explicit warmup and timed loops driven by `BenchmarkRunner` iteration settings.
- GPU timing comes from `VulkanContext::measure_gpu_time_ms(...)`.
- Emits:
  - one `BenchmarkResult` per `(layout, particle_count)`
  - one `BenchmarkMeasurementRow` per timed iteration
  - correctness pass/fail status per row

## Resource Lifecycle
- Builds AoS and SoA resources independently.
- Cleans up pipelines, descriptor objects, shader modules, and buffers explicitly.
- Returns empty result set on setup failures.

## Preconditions
- Requires GPU timestamp support.
- Requires both `06_aos.comp.spv` and `06_soa.comp.spv` (or explicit user paths).
