# Experiment 22: Prefix Sum or Scan

## 1. Focus
- Measure block-scan performance and the effect of batching multiple items per thread.

## 2. Question
- What is the best `items_per_thread` tradeoff for the shared-memory scan baseline?

## 3. Variants
- `items_per_thread_1`
- `items_per_thread_4`
- `items_per_thread_8`

## 4. Method
- Run the same inclusive-scan kernel shape while changing only how many items each thread handles.
- Keep output semantics, correctness rules, and timing policy fixed across the sweep.

## 5. Outputs
- Median GPU time by `items_per_thread`.
- Throughput by configuration.
- Practical default for later scan and compaction work.

## 6. Interpretation
- More work per thread is not automatically better if it reduces occupancy or inflates local state.
- This is the shared-memory scan baseline that later subgroup variants should be compared against.
