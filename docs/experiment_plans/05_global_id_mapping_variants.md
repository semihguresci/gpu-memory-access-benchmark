# Experiment 05: Global ID Mapping Variants

## 1. Focus
- Compare common indexing formulas used to map global IDs onto logical work.

## 2. Question
- What overhead is introduced by offset and grid-stride indexing relative to direct mapping?

## 3. Variants
- `direct`
- `fixed_offset`
- `grid_stride`

## 4. Method
- Keep arithmetic, memory traffic, and output semantics constant.
- Change only the mapping from invocation ID to logical element and measure the result across larger workloads.

## 5. Outputs
- Median GPU time by mapping strategy.
- Throughput by mapping strategy.
- Relative overhead of flexible indexing vs direct mapping.

## 6. Interpretation
- Grid-stride loops buy scalability and launch flexibility, but the control-flow cost still needs to be measured.
- Direct mapping remains the simplest reference path for kernels that can launch exact coverage.
