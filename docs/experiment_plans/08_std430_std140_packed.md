# Experiment 08: std430 vs std140 vs Packed

## 1. Focus
- Separate API-driven layout padding from useful logical payload in storage buffers.

## 2. Question
- How much cost comes from `std140` and `std430` padding compared with a tightly packed representation?

## 3. Variants
- `packed`
- `std430`
- `std140`

## 4. Method
- Keep the same logical fields and arithmetic while changing only the storage layout.
- Measure both runtime and physical bytes per record so padding cost stays visible in the analysis.

## 5. Outputs
- Median GPU time by layout.
- Physical storage bytes per record.
- Useful-payload GB/s and relative slowdown vs `packed`.

## 6. Interpretation
- Padded layouts can simplify interoperability, but they should not be mistaken for free memory traffic.
- The important comparison is useful work delivered per unit time, not just raw storage stride.
