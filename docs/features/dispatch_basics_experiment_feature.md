# Dispatch Basics Experiment Feature (01)

## Source
- `src/experiments/dispatch_basics_experiment.cpp`

## Purpose
Establishes a correctness-first compute baseline with explicit upload, dispatch, and readback timing phases.

## Variants
- `contiguous_write`: writes index values to storage buffer.
- `noop`: dispatch overhead baseline with no data mutation.

## Sweep Dimensions
- Problem sizes: powers of two from `2^10` to `2^24` (clamped by scratch size and device dispatch limits).
- Dispatch counts: `1, 4, 16, 64, 128, 256, 512, 1024`.
- Local size: `64` threads in X.

## GPU Resource Model
- Device-local storage buffer.
- Host-visible upload staging buffer.
- Host-visible readback staging buffer.
- Separate pipeline setup for write and no-op variants.

## Per-Point Execution
1. Fill upload staging with sentinel value.
2. Upload to device buffer.
3. Dispatch compute workload (possibly repeated by dispatch count).
4. Read back device buffer.
5. Validate correctness.
6. Record detailed measurement row and summary sample.

## Metrics and Outputs
- Per-iteration rows include:
  - `gpu_ms` (dispatch stage)
  - end-to-end ms
  - throughput and effective GB/s
  - correctness flag and notes
- Summary results emitted per `(variant, problem_size, dispatch_count)`.

## Correctness Contract
- Any failed row marks `all_points_correct = false`.
- Non-finite stage timings and content mismatches are tracked in row notes.

## Failure Conditions
- Missing timestamp support.
- Missing shaders.
- Insufficient scratch size.
- Vulkan resource creation or memory mapping failures.
