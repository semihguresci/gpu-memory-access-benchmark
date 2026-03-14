# GPU Timestamp Timer Utility Feature

## Source
- `include/utils/gpu_timestamp_timer.hpp`
- `src/utils/gpu_timestamp_timer.cpp`

## Purpose
Encapsulates Vulkan timestamp query pool operations for command-buffer-level GPU timing.

## Core Operations
- `initialize(...)`: creates timestamp query pool.
- `record_start(...)`: resets query pool and writes start timestamp.
- `record_end(...)`: writes end timestamp.
- `resolve_milliseconds(...)`: fetches query results and converts to ms.
- `shutdown(...)`: destroys query pool and resets internal state.

## Validation Rules
- Requires valid query pool and positive timestamp period.
- Rejects invalid query results (including end < start).
- Returns failure if result is not finite.

## Notes
- This utility lives in `VulkanComputeUtils` namespace.
- `VulkanContext::measure_gpu_time_ms(...)` uses this utility as its active runtime timing implementation.
