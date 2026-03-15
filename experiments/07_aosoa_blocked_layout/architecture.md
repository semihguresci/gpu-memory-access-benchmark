# Experiment 07 AoSoA Blocked Layout: Runtime Architecture

## Objective
Benchmark three layout families under the same particle update kernel:
- AoS
- SoA
- AoSoA blocked (`4, 8, 16, 32`)

Workload payload:
- 16 floats per particle (`pos/vel/mass/dt`, 4 hot auxiliary fields, 4 cold fields)
- Kernel updates hot fields; cold fields remain unchanged and are validated

## Data Flow
1. Host allocates and maps variant-specific storage buffers.
2. Host seeds deterministic particle state per variant.
3. Compute dispatch runs one update step (`pos += vel * dt`, `mass += epsilon`).
4. Host validates GPU results against deterministic CPU reference.
5. Row metrics and summary statistics are exported to benchmark JSON.

AoSoA storage order is block-major (`block -> field -> lane`) so block-size changes materially alter memory layout.

## Vulkan Resources
- Shared context: one compute queue, timestamp timer, and command buffer from `VulkanContext`.
- AoS:
  - one storage buffer
  - one descriptor set layout + descriptor set
  - one compute pipeline (`07_aos.comp`)
- SoA:
  - eight storage buffers
  - one descriptor set layout + descriptor set
  - one compute pipeline (`07_soa.comp`)
- AoSoA:
  - one blocked storage buffer
  - one descriptor set layout + descriptor set
  - one compute pipeline (`07_aosoa_blocked.comp`)

## Push Constants
- AoS/SoA: `count`
- AoSoA: `count`, `block_size`, `block_count`

## Measurement Contract
- Primary timing: GPU dispatch ms from timestamp queries.
- Supporting timing: end-to-end host measured per iteration.
- Row schema fields include:
  - `experiment_id`
  - `variant`
  - `problem_size`
  - `iteration`
  - `gpu_ms`
  - `throughput`
  - `gbps`
  - `correctness_pass`
  - `notes`

## Teardown Policy
All Vulkan objects are destroyed in reverse creation order and handles are reset to `VK_NULL_HANDLE`.
