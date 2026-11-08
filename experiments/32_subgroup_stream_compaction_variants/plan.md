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
- The ballot path assigns input positions from dense active subgroup/lane ranks, preserving logical order without assuming a local-ID mapping. Active counts handle partial subgroups and shared arrays support up to 256 subgroups.
- Every measured iteration validates all block counts, survivors and untouched output tails.

## 5. Outputs
- Median GPU time by valid ratio.
- Speedup of subgroup ballot vs shared atomic append.
- Shader payload GB/s: `(2 * valid_count + block_count) * 4` bytes for survivor-only input loads, survivor stores and one count per block. This excludes shared atomics and is not measured DRAM traffic.

## 6. Interpretation
- This isolates subgroup compaction mechanics without the added cost of a global scan pipeline.
- Ordering guarantees should be interpreted separately from raw throughput.
- The portable ballot path's membership setup and barriers are timed; its stability contract is stronger than the unordered atomic path's contract.
