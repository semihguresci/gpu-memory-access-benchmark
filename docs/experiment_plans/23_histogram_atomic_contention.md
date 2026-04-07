# Experiment 23: Histogram and Atomic Contention

## 1. Focus
- Measure how atomic-heavy histogram construction responds to input skew and privatization.

## 2. Question
- How much does shared privatization reduce contention relative to direct global atomics?

## 3. Variants
- `global_atomics`
- `privatized_shared`
- input distributions `uniform`, `mixed_hotset`, and `hot_bin_90`

## 4. Method
- Build the same histogram from deterministic inputs under several contention regimes.
- Compare direct global updates against a workgroup-private accumulation path with controlled flush behavior.

## 5. Outputs
- Median GPU time by variant and distribution.
- Speedup of `privatized_shared` vs `global_atomics`.
- Contention sensitivity across the input distributions.

## 6. Interpretation
- Histogram performance is dominated by contention shape, not just by the number of input elements.
- Privatization should be judged by both its best-case speedup and its cost on low-contention inputs.
