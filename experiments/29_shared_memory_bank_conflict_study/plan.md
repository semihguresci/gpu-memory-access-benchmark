# Experiment 29: Shared Memory Bank Conflict Study

## 1. Focus
- Measure how shared-memory stride changes throughput because of bank conflicts.

## 2. Question
- How severe is the slowdown from conflict-heavy strides, and how much does padding recover?

## 3. Variants
- `stride_1`
- `stride_2`
- `stride_4`
- `stride_8`
- `stride_16`
- `stride_32`
- `padded_fix`

## 4. Method
- Load one workgroup tile into shared memory and reread it with a configurable stride.
- Keep global-memory payload fixed so the difference is dominated by on-chip behavior.

## 5. Outputs
- Median GPU time by stride.
- Slowdown relative to `stride_1`.
- Padding recovery relative to `stride_32`.

## 6. Interpretation
- Shared memory is not automatically fast; bank conflicts can erase the intended gain.
- Padding should be justified by a measured recovery, not by habit.
