# Experiment 21: Parallel Reduction

## 1. Focus
- Compare baseline and hierarchical reduction strategies for a classic GPU primitive.

## 2. Question
- When does shared-memory tree reduction beat a simpler global-atomic approach?

## 3. Variants
- `global_atomic`
- `shared_tree`

## 4. Method
- Reduce the same deterministic input values with the same final operator in both variants.
- Sweep problem size across the available scratch budget and dispatch limit.

## 5. Outputs
- Median GPU time by variant and size.
- Speedup of `shared_tree` vs `global_atomic`.
- Effective input-read throughput.

## 6. Interpretation
- Hierarchical reduction usually pays off only after the workload is large enough to amortize setup overhead.
- This establishes the pre-subgroup baseline for later reduction work.
