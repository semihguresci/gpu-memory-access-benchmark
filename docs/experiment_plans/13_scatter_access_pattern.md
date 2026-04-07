# Experiment 13: Scatter Access Pattern

## 1. Focus
- Measure the cost of indirect writes and the slowdown introduced by target collisions.

## 2. Question
- How much do write-target distribution and contention change scatter throughput?

## 3. Variants
- `unique_targets`
- `low_collision_random`
- `high_collision_clustered`

## 4. Method
- Keep the number of logical writes and arithmetic work fixed while changing only the target-index distribution.
- Validate the final output against deterministic CPU reference behavior for each collision regime.

## 5. Outputs
- Median GPU time by scatter distribution.
- Relative slowdown vs the unique-target baseline.
- Contention-sensitive throughput comparison.

## 6. Interpretation
- Scatter is not just the write-side mirror of gather because collisions can serialize progress.
- The result should guide whether later pipelines need privatization, staging, or compaction.
