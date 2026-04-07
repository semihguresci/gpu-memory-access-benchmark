# Experiment 11: Coalesced vs Strided Access

## 1. Focus
- Measure the direct cost of breaking contiguous lane access into wider strides.

## 2. Question
- How quickly does throughput fall as address stride increases and coalescing quality drops?

## 3. Variants
- `stride_1`
- `stride_2`
- `stride_4`
- `stride_8`
- `stride_16`

## 4. Method
- Keep arithmetic, output count, and useful bytes fixed while changing only the source index stride.
- Run the sweep at sizes that are large enough to stay in the bandwidth-bound region.

## 5. Outputs
- Median GPU time by stride.
- Throughput and effective GB/s by stride.
- Slowdown curve relative to `stride_1`.

## 6. Interpretation
- Coalescing is a first-order performance rule on bandwidth-bound kernels.
- The resulting curve is a concrete demonstration of why strided access wastes transactions and cache lines.
