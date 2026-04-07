# Experiment 02: Local Size Sweep

## 1. Focus
- Measure how `local_size_x` changes occupancy, scheduling behavior, and throughput.

## 2. Question
- Which legal workgroup size is fastest and most stable for a fixed contiguous kernel on the tested GPU?

## 3. Variants
- `local_size_x = 32, 64, 128, 256, 512, 1024`
- `contiguous_write`
- `noop`

## 4. Method
- Run the same contiguous kernel across all legal workgroup sizes on the device.
- Keep arithmetic, memory layout, timing path, and correctness checks unchanged between points.

## 5. Outputs
- Median and p95 GPU time by `local_size_x`.
- Throughput by `local_size_x`.
- Recommended default local size for later experiments.

## 6. Interpretation
- The winner is architecture-specific and should be treated as a platform default, not a universal rule.
- The no-op path helps separate launch behavior from memory-work behavior.
