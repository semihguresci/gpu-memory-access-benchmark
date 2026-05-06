# Experiment 06: AoS vs SoA

## 1. Focus
- Measure how record layout changes memory efficiency for field-oriented GPU work.

## 2. Question
- Does struct-of-arrays outperform array-of-structs for the same logical particle update?

## 3. Variants
- `aos`
- `soa`

## 4. Method
- Run the same logical kernel over identical particle data in both layouts.
- Keep the fields touched, arithmetic, and output contract fixed across both variants.

## 5. Outputs
- Median GPU time by layout.
- Useful-payload GB/s by layout.
- Speedup of `soa` relative to `aos`.

## 6. Interpretation
- Field-wise access usually favors SoA because threads read contiguous values from the same field.
- Any AoS win would need to be explained by a different access shape or reduced address-generation cost.
