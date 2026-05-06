# Experiment 26: Warp-Level Coalescing Alignment

## 1. Focus
- Measure the cost of misalignment when lane access remains otherwise contiguous.

## 2. Question
- How much performance is lost when a warp or wave crosses worse transaction boundaries because of base offset?

## 3. Variants
- alignment offsets `0`, `4`, `8`, `16`, `32`, `64`

## 4. Method
- Keep the same contiguous lane pattern and shift only the base address alignment.
- Measure the same useful work at each offset with deterministic validation.

## 5. Outputs
- Median GPU time by alignment offset.
- Effective GB/s by offset.
- Slowdown relative to the aligned baseline.

## 6. Interpretation
- Contiguous access is not automatically optimal if alignment expands the transaction footprint.
- This experiment turns alignment into a measurable rule instead of a vague best practice.
