# Experiment 14: Read Reuse and Cache Locality

## 1. Focus
- Measure how temporal reuse distance changes read performance on the same logical data.

## 2. Question
- How quickly does the benefit of a second read decay as more work is inserted between touches?

## 3. Variants
- `reuse_distance_1`
- `reuse_distance_32`
- `reuse_distance_256`
- `reuse_distance_4096`
- `reuse_distance_full_span`

## 4. Method
- Build deterministic index schedules where each source element is read twice with a controlled gap.
- Keep total reads, writes, and arithmetic fixed while varying only the reuse distance.

## 5. Outputs
- Median GPU time by reuse distance.
- Speedup relative to the far-distance baseline.
- Stability of the locality benefit across the reuse sweep.

## 6. Interpretation
- This experiment estimates a practical locality window rather than a literal cache size.
- It helps explain when reordering work or staging data is likely to matter.
