# Experiment 03: Memory Copy Baseline

## 1. Focus
- Characterize simple read, write, and copy kernels as a practical memory-throughput baseline.

## 2. Question
- How much bandwidth does the GPU sustain for the simplest contiguous memory modes?

## 3. Variants
- `read_only`
- `write_only`
- `read_write_copy`

## 4. Method
- Run one simple contiguous kernel per memory mode across a size sweep.
- Time dispatch separately from host transfers and validate each mode against a deterministic reference.

## 5. Outputs
- Median GPU time by mode and size.
- Effective GB/s by mode.
- Saturation knee and steady-state comparison across modes.

## 6. Interpretation
- This is the closest thing to a roofline-style denominator for later bandwidth claims.
- More complex kernels should be compared against this baseline before claiming efficiency.
