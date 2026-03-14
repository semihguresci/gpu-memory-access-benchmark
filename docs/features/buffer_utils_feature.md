# Buffer Resource Management Feature

## Source
- `src/utils/buffer_utils.cpp`

## Purpose
Provides explicit Vulkan buffer allocation and teardown helpers for experiments.

## API Behavior
- `create_buffer_resource(...)`
  - Creates `VkBuffer`
  - Queries memory requirements
  - Selects memory type matching required properties
  - Allocates and binds `VkDeviceMemory`
- `destroy_buffer_resource(...)`
  - Destroys buffer
  - Frees memory
  - Clears struct fields

## Memory Type Selection
- Uses physical device memory properties.
- Matches both type bitmask and requested property flags.
- Returns failure when no compatible memory type exists.

## Safety Guarantees
- Cleans up partially created Vulkan objects on failure paths.
- Leaves output resource in a known state (`VK_NULL_HANDLE` and size `0`) after destroy.

## Typical Usage
- Device-local storage buffers for compute workloads.
- Host-visible staging/readback buffers for upload and result validation.
