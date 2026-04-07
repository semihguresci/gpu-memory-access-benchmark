# Experiment 31: Subgroup Scan Variants

## 1. Focus
- Compare shared-memory block scan against subgroup-assisted block scan.

## 2. Question
- How much workgroup-local scan overhead can subgroup intrinsics remove?

## 3. Variants
- `shared_block_scan`
- `subgroup_block_scan`
- `items_per_thread = 1, 4, 8`

## 4. Method
- Run an independent inclusive scan per workgroup block for both implementations.
- Keep logical count and output semantics fixed while sweeping `items_per_thread`.

## 5. Outputs
- Median GPU time by `items_per_thread`.
- Speedup of subgroup scan vs shared scan.
- Stability comparison for both implementations.

## 6. Interpretation
- This isolates the block-local primitive rather than a full hierarchical global scan.
- Any win should be interpreted as a building block improvement for larger scan pipelines.
