# Vulkan Context Feature

## Source
- `include/vulkan_context.hpp`
- `src/vulkan_context.cpp`

## Purpose
Owns Vulkan lifetime and shared command/timestamp resources used by compute experiments.

## Owned Resources
- `VkInstance`
- optional debug messenger (`VK_EXT_debug_utils`)
- selected `VkPhysicalDevice`
- `VkDevice` + compute queue
- command pool + primary command buffer
- submit fence
- `VulkanComputeUtils::GpuTimestampTimer`

## Initialization Sequence
1. Create instance (optionally with validation layer and debug utils extension).
2. Create debug messenger when validation is enabled.
3. Pick physical device with compute-capable queue family.
4. Create logical device and fetch compute queue.
5. Create command pool, command buffer, and fence.
6. Probe timestamp support and initialize `GpuTimestampTimer` when available.

## Timing API
`measure_gpu_time_ms(record_commands)`:
- Resets fence and command pool.
- Records command buffer.
- Delegates timestamp begin/end writes to `GpuTimestampTimer`.
- Submits and waits.
- Delegates timestamp resolution and milliseconds conversion to `GpuTimestampTimer`.
- Returns `NaN` on failure.

## Shutdown Rules
- Waits for device idle.
- Destroys resources in reverse order.
- Resets Vulkan handles to `VK_NULL_HANDLE`.
- Resets capability/state flags.

## Selection Policy
- Picks first physical device with compute queue support.
- Timestamp capability is validated through queue family `timestampValidBits`.
