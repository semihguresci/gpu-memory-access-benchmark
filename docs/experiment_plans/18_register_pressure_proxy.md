# Experiment 18: Register Pressure Proxy Study

## 1. Focus
- Approximate the cost of rising register pressure without changing the overall memory pattern.

## 2. Question
- How much performance is lost as the kernel carries more live temporaries?

## 3. Variants
- `temp_4`
- `temp_8`
- `temp_16`
- `temp_32`

## 4. Method
- Keep the same memory traffic and output contract while increasing the amount of per-thread temporary state.
- Validate every variant against the same deterministic CPU reference.

## 5. Outputs
- Median GPU time by temporary-count variant.
- Slowdown relative to the lightest register-pressure path.
- Evidence of occupancy or scheduling cliffs.

## 6. Interpretation
- This is a proxy experiment, so it should be read as pressure sensitivity rather than a literal register count measurement.
- Large slowdowns indicate that arithmetic-only optimizations can still fail if they inflate live state.
