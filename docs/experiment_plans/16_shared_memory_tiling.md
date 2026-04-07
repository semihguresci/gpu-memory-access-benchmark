# Experiment 16: Shared or Workgroup Memory Tiling

## 1. Focus
- Compare direct global reads against explicit workgroup-memory tiling for a stencil-like workload.

## 2. Question
- When does local staging outperform direct global access once halo loads and barriers are included?

## 3. Variants
- `direct_global`
- `shared_tiled`
- reuse-radius sweep

## 4. Method
- Use the same 1D sliding-window sum for both implementations.
- Sweep neighborhood radius while keeping output count, arithmetic, and validation rules fixed.

## 5. Outputs
- Median GPU time by variant and radius.
- Speedup of `shared_tiled` vs `direct_global`.
- Shared-memory footprint and estimated global-traffic reduction.

## 6. Interpretation
- Shared memory only wins when reuse is large enough to amortize halo loading and barrier cost.
- This is the baseline for all later shared-memory tuning experiments.
