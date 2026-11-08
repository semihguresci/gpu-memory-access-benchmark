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
- Compare stable three-stage compaction with unordered global atomic append; these intentionally have different ordering contracts, recorded in row notes.
- Report input elements/s separately from estimated source-level storage-buffer GB/s. For `N` input elements, `V` survivors, and `B` workgroups, atomic append requests `4 * (4V)` bytes: survivor input/output plus one 8-byte modeled counter read-modify-write per survivor.
- The three-stage estimate is `4 * (N + 4V + 4B + 1)` bytes: stage 1 writes `N` prefixes and `B` counts; stage 2 reads `2B` counts and writes `B` block prefixes plus one total; scatter reads input/local-prefix/block-prefix and writes output only for the `V` survivors. Repeated accesses are source operations, not measured DRAM transfers.

## 5. Outputs
- Median GPU time by keep ratio.
- Input elements/s and estimated total storage-buffer GB/s, including auxiliary traffic.
- Stage-sensitive interpretation of where compaction cost shifts across sparsity regimes.

## 6. Interpretation
- Stream compaction cost is shaped by both the scan path and the amount of output that survives.
- This is the baseline for later subgroup-assisted compaction work.
