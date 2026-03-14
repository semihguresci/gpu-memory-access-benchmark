# Vulkan Compute Helper Feature

## Source
- `src/utils/vulkan_compute_utils.cpp`

## Purpose
Offers low-level reusable helpers for compute pipeline setup, SPIR-V loading, descriptor updates, and synchronization barriers.

## File and Shader Helpers
- Reads binary files for SPIR-V modules.
- Creates shader modules from bytecode.
- Resolves shader paths across common build/runtime directories.

## Descriptor and Pipeline Helpers
- Create descriptor set layouts.
- Create descriptor pools.
- Allocate descriptor sets.
- Update descriptor buffers.
- Create pipeline layouts (with and without push constants).
- Create compute pipelines.

## Dispatch Utilities
- `compute_group_count_1d(element_count, local_size_x)` for 1D launch sizing.

## Synchronization Utilities
- Transfer write -> compute read/write barrier.
- Compute write -> transfer read barrier.

## Design Notes
- Functions are intentionally explicit wrappers over Vulkan calls.
- Callers own all created Vulkan objects and must destroy them.
