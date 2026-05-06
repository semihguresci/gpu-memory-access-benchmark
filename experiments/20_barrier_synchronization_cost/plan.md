# Experiment 20: Barrier and Synchronization Cost

## 1. Focus
- Isolate the runtime cost of workgroup barriers and synchronization placement.

## 2. Question
- How much overhead comes from barriers themselves, and does that cost depend on how work is tiled?

## 3. Variants
- `flat_loop_no_barrier`
- `tiled_regions_no_barrier`
- `flat_loop_with_barrier`
- `tiled_regions_with_barrier`

## 4. Method
- Use the same logical output and arithmetic while changing whether the kernel runs as a flat loop or staged tiled regions.
- Add or remove barriers without changing the final output contract.

## 5. Outputs
- Median GPU time by synchronization strategy.
- Barrier overhead relative to the no-barrier forms.
- Placement sensitivity between flat and tiled execution shapes.

## 6. Interpretation
- A barrier cost is only meaningful relative to the work it protects.
- This experiment explains why some shared-memory kernels fail even when their memory traffic looks favorable on paper.
