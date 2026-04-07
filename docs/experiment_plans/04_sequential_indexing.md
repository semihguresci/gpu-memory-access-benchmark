# Experiment 04: Sequential Indexing

## 1. Focus
- Establish the good-path baseline for fully contiguous indexing without indirection or stride.

## 2. Question
- How fast is a simple sequential read-modify-write kernel before access-pattern penalties are introduced?

## 3. Variants
- `sequential_indexing`
- problem-size sweep

## 4. Method
- Map one invocation to one contiguous element in both source and destination buffers.
- Keep arithmetic and memory footprint fixed while scaling only the logical workload size.

## 5. Outputs
- Median GPU time.
- Throughput and effective GB/s.
- Baseline reference for later access-pattern experiments.

## 6. Interpretation
- This is the comparison point for gather, scatter, stride, and reuse studies.
- Later slowdowns should be explained relative to this contiguous baseline, not in isolation.
