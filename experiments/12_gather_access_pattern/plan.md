# Experiment 12: Gather Access Pattern

## 1. Focus
- Measure the penalty of indirect indexed reads relative to sequential access.

## 2. Question
- How much does gather performance depend on index coherence and locality?

## 3. Variants
- `sequential`
- `block_coherent`
- `random`
- `clustered_random`

## 4. Method
- Read from a source buffer through a host-generated index buffer with controlled distributions.
- Keep output writes and arithmetic fixed so the result is dominated by read-side access behavior.

## 5. Outputs
- Median GPU time by index distribution.
- Relative slowdown vs the sequential baseline.
- Throughput and useful-payload GB/s.

## 6. Interpretation
- Gather cost should track how much locality survives the indirection pattern.
- This is the baseline for later systems that depend on neighbor lookups, sparse reads, or visibility indirection.
