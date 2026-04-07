# Experiment 10: Scalar Type Width Sweep

## 1. Focus
- Test the tradeoff between narrower storage formats and the cost of packing, unpacking, and reduced precision.

## 2. Question
- When does narrower storage become a net performance win over the `fp32` baseline?

## 3. Variants
- `fp32`
- `fp16_storage`
- `u32`
- `u16`
- `u8`

## 4. Method
- Run the same logical scalar update across several storage representations.
- Use tolerance-aware validation so runtime gains can be interpreted together with numerical loss.

## 5. Outputs
- Median GPU time by representation.
- Throughput and effective GB/s.
- Maximum and mean error for reduced-precision variants.

## 6. Interpretation
- Narrower formats only help when bandwidth savings exceed conversion overhead and precision cost.
- Portability and feature support matter as much as raw speed for any representation that is not plain `fp32`.
