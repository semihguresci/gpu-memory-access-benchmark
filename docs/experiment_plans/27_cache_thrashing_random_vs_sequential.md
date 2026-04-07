# Experiment 27: Cache Thrashing, Random vs Sequential

## 1. Focus
- Compare a locality-friendly sequential pass against a cache-thrashing random pass.

## 2. Question
- How much throughput is lost when the same useful work is reordered into a poor-locality access pattern?

## 3. Variants
- `sequential`
- `random`

## 4. Method
- Run the same logical transform with identical output semantics and only change the access ordering.
- Keep arithmetic and total payload fixed so the result isolates locality loss rather than extra work.

## 5. Outputs
- Median GPU time by ordering.
- Effective GB/s by ordering.
- Slowdown of `random` relative to `sequential`.

## 6. Interpretation
- This is a direct locality study, so the important result is the gap between the two orderings.
- The measured penalty supports arguments for sorting, binning, or data-layout reordering.
