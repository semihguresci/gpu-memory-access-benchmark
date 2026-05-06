# Experiment 07: AoSoA or Blocked Layout

## 1. Focus
- Evaluate blocked record layouts that sit between pure AoS and pure SoA.

## 2. Question
- Can AoSoA recover some of SoA's access efficiency without fully splitting every field into its own array?

## 3. Variants
- `aos`
- `aosoa_blocked`
- `soa`

## 4. Method
- Keep the logical particle update fixed and compare interleaved, blocked, and fully split storage.
- Use the same workload size, timing path, and correctness rules for all layouts.

## 5. Outputs
- Median GPU time by layout.
- Relative speedup vs `aos`.
- Practical guidance on whether blocked layouts justify their complexity.

## 6. Interpretation
- AoSoA is a compromise layout, so the result should be read as a tradeoff study rather than a binary winner-take-all test.
- If a blocked layout wins, it may be the best engineering choice when full SoA is intrusive.
