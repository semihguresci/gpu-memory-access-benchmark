# Experiment 25: Spatial Binning or Clustered Culling Capstone

## 1. Focus
- Combine earlier memory and primitive lessons into a rendering-adjacent compute pipeline.

## 2. Question
- Which list-building strategy performs best across sparse, dense, and clustered scene distributions?

## 3. Variants
- `naive_append`
- `local_staging`
- `compacted_lists`
- coherent and incoherent scene ordering

## 4. Method
- Build bin or cluster membership lists from the same deterministic scene generator.
- Compare pipeline variants under several scene distributions while keeping object counts and correctness rules fixed.

## 5. Outputs
- Total GPU time by pipeline variant.
- Stage breakdown for list construction.
- Practical winner by scene distribution.

## 6. Interpretation
- This is an end-to-end engineering experiment, not a pure microbenchmark.
- A good result is one that explains which earlier primitive or access-pattern choice mattered in the final pipeline.
