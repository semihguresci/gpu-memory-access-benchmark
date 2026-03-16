# Experiment 09 vec3 vs vec4 Padding Costs: Runtime Architecture

## Objective
Measure the cost of layout shape and padding with equivalent per-particle update logic across three variants:
- `vec3_padded`: AoS struct with `vec3` members that induce alignment padding
- `vec4`: AoS struct packing scalar companions into `vec4.w`
- `split_scalars`: SoA-style scalar arrays in one storage buffer

Logical payload per particle:
- `11` floats (`coeffs[3]`, `position[3]`, `velocity[3]`, `mass`, `dt`) => `44` logical bytes

Variant storage stride:
- `vec3_padded`: `64` bytes
- `vec4`: `48` bytes
- `split_scalars`: `44` bytes

## Data Flow
1. Host allocates one mapped storage buffer per variant.
2. Host seeds deterministic particle state in variant-specific representation.
3. Compute dispatch runs one update step per particle.
4. Host validates GPU outputs against deterministic CPU expected values.
5. Per-iteration rows and case summaries are exported to JSON.

## Vulkan Resources
- Shared context: compute queue + timestamp timer from `VulkanContext`
- Per variant:
  - one storage buffer
  - one descriptor set layout + descriptor set
  - one compute pipeline

## Push Constants
- `count` (particle count for current dispatch)

## Measurement Contract
- Primary timing: GPU dispatch ms via timestamp queries
- Supporting timing: end-to-end host iteration time
- Row fields:
  - `experiment_id`
  - `variant`
  - `problem_size`
  - `iteration`
  - `gpu_ms`
  - `throughput`
  - `gbps`
  - `correctness_pass`
  - `notes`

`notes` include:
- `storage_bytes_per_particle`
- `logical_bytes_per_particle`
- `alignment_waste_ratio`

## Teardown Policy
All Vulkan objects are destroyed in reverse creation order and handles are reset to `VK_NULL_HANDLE`.
