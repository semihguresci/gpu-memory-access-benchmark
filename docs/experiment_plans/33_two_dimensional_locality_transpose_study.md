# Experiment 33: 2D Locality and Transpose Study

## 1. Focus
- Move beyond 1D buffers into matrix-shaped access and transpose behavior.

## 2. Question
- How much does tiling recover from the poor locality of naive transpose?

## 3. Variants
- `row_major_copy`
- `naive_transpose`
- `tiled_transpose`
- `tiled_transpose_padded`

## 4. Method
- Use a square matrix sized from the scratch budget.
- Keep input data and output semantics fixed while comparing naive and tiled access patterns.

## 5. Outputs
- Median GPU time by variant.
- Effective GB/s by variant.
- Speedup of tiled transpose over naive transpose.

## 6. Interpretation
- This experiment connects the project's memory story to image-style and rendering-adjacent workloads.
- The padded tiled variant shows whether bank-conflict mitigation matters in the transpose case.
