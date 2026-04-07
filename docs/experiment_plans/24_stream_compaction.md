# Experiment 24: Stream Compaction

## 1. Focus
- Measure the baseline compaction pipeline across several keep ratios.

## 2. Question
- How does valid-element ratio change the cost of filtering and compact-writing a stream?

## 3. Variants
- `keep_5`
- `keep_25`
- `keep_50`
- `keep_75`
- `keep_95`

## 4. Method
- Run the same compaction pipeline on deterministic inputs with different valid ratios.
- Keep output semantics fixed and validate counts and compacted values after each run.

## 5. Outputs
- Median GPU time by keep ratio.
- Effective compacted-elements throughput.
- Stage-sensitive interpretation of where compaction cost shifts across sparsity regimes.

## 6. Interpretation
- Stream compaction cost is shaped by both the scan path and the amount of output that survives.
- This is the baseline for later subgroup-assisted compaction work.
