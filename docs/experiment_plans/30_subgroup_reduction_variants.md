# Experiment 30: Subgroup Reduction Variants

## 1. Focus
- Extend the reduction family with subgroup-assisted execution.

## 2. Question
- Does subgroup reduction beat the classic shared-tree reduction for this one-pass reduction pattern?

## 3. Variants
- `shared_tree`
- `subgroup_hybrid`

## 4. Method
- Reduce one element per thread, then combine per-workgroup results with the same final accumulation step.
- Sweep several problem sizes up to the available scratch budget and dispatch limit.

## 5. Outputs
- Median GPU time by problem size.
- Speedup of `subgroup_hybrid` vs `shared_tree`.
- Effective input-read throughput.

## 6. Interpretation
- This isolates subgroup benefit in the reduction primitive without changing the global algorithm shape.
- The result should be read against Experiment 21, not as a replacement for full multi-pass reduction design.
