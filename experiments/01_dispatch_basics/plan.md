# Experiment 01: Dispatch Basics

## 1. Focus
- Establish the minimum viable Vulkan compute benchmark and validation path.

## 2. Question
- Can the project execute upload, dispatch, and readback with correct outputs and stable GPU timing?

## 3. Variants
- `contiguous_write`
- `noop`
- problem-size and dispatch-count sweep

## 4. Method
- Use a deterministic 1D write kernel and a no-op control path.
- Sweep problem size and dispatch count while keeping timing and synchronization policy fixed.

## 5. Outputs
- Median GPU dispatch time.
- End-to-end time.
- Throughput and correctness pass rate.

## 6. Interpretation
- This is a harness-validation experiment first and a performance result second.
- Later experiments should only be trusted if they remain consistent with this baseline.
