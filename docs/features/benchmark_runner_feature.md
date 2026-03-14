# Benchmark Runner Feature

## Source
- `src/benchmark_runner.cpp`

## Purpose
Provides reusable benchmark execution loops and summary statistics for timed samples.

## What It Does
- Runs warmup iterations before recording samples.
- Supports two modes:
  - host wall-time timing (`run`)
  - externally measured timing (`run_timed`, used for GPU timestamp timings)
- Produces stable summary metrics through `summarize_samples`.

## Output Metrics
- `average_ms`
- `min_ms`
- `max_ms`
- `median_ms` (50th percentile, linearly interpolated)
- `p95_ms` (95th percentile, linearly interpolated)
- `sample_count` (finite samples only)

## Robustness Rules
- Non-finite samples are filtered out before summary computation.
- If no finite samples exist, all reported metrics are `NaN`.
- Sample vectors pre-reserve capacity based on configured iteration count.

## Integration Points
- Consumed by experiment implementations to standardize result format.
- Works with `VulkanContext::measure_gpu_time_ms(...)` through `run_timed`.
