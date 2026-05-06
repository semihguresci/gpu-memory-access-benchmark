# Experiment 09: vec3, vec4, and Padding Costs

## 1. Focus
- Compare common vector-friendly storage choices against tighter scalarized layouts.

## 2. Question
- How much performance is lost when `vec3` and `vec4`-style layouts carry padding that the kernel does not need?

## 3. Variants
- `split_scalars`
- `vec3_padded`
- `vec4`

## 4. Method
- Use the same logical per-record values and the same arithmetic in every variant.
- Change only the storage representation and validate all outputs against the same CPU reference.

## 5. Outputs
- Median GPU time by layout.
- Useful-payload GB/s by layout.
- Padding overhead relative to the scalarized baseline.

## 6. Interpretation
- Vector convenience and alignment hygiene can cost real bandwidth when the shader does not use the extra padded bytes.
- This experiment helps decide when explicit scalar packing is worth the added code complexity.
