# Experiment 32: Subgroup Stream Compaction Variants

## 1. Focus
- Compare per-workgroup compaction built on shared atomics versus subgroup ballot ranking.

## 2. Question
- How much does subgroup ballot reduce the overhead of local compaction across sparsity regimes?

## 3. Variants
- `shared_atomic_block`
- `subgroup_ballot`
- valid ratios `5%`, `25%`, `50%`, `75%`, `95%`

## 4. Method
- Compact each workgroup into its own fixed output segment.
- Validate exact counts for both variants and stable ordering only for the subgroup-ballot path.

## 5. Outputs
- Median GPU time by valid ratio.
- Speedup of subgroup ballot vs shared atomic append.
- Effective payload GB/s using actual valid-count writes.

## 6. Interpretation
- This isolates subgroup compaction mechanics without the added cost of a global scan pipeline.
- Ordering guarantees should be interpreted separately from raw throughput.
