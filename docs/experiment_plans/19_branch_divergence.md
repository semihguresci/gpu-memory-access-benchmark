# Experiment 19: Branch Divergence

## 1. Focus
- Measure how different predicate distributions affect warp or wave efficiency.

## 2. Question
- How much slowdown comes from divergent control flow when useful work stays otherwise similar?

## 3. Variants
- `uniform_true`
- `alternating`
- `random_p25`
- `random_p50`
- `random_p75`

## 4. Method
- Run the same kernel body with controlled branch masks and deterministic predicate generation.
- Keep writes and arithmetic comparable so the dominant change is branch coherence.

## 5. Outputs
- Median GPU time by branch pattern.
- Slowdown relative to the uniform baseline.
- Divergence sensitivity across several predicate mixes.

## 6. Interpretation
- Divergence penalties depend on pattern shape, not just branch probability.
- This experiment is strongest when read together with memory-behavior experiments that also vary access coherence.
