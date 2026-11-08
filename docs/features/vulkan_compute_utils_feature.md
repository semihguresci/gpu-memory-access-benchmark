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
- Validate per-stage and descriptor-set storage-buffer descriptor limits before resource creation.
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
- Compute write -> host read barriers for one buffer or all shader-written memory.
- Transfer write -> host read barrier for a staging-buffer range.

Host-read barriers establish memory visibility; callers still wait for GPU completion before accessing mapped output. Direct-output experiments include the visibility dependency in their GPU interval and export its timing scope. Staged paths include it in readback instead.

## Design Notes
- Functions are intentionally explicit wrappers over Vulkan calls.
- Callers own all created Vulkan objects and must destroy them.
