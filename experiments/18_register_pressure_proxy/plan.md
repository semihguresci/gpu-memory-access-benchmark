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
- Evaluate exactly 32 independent uint32 chains, each initialized once, updated for eight rounds, and folded once into the same output in chain order.
- Schedule these chains in batches of 4, 8, 16, or 32 live states. Every variant performs 256 chain updates per element, reads one input word, and writes one output word.
- Validate every variant against the same deterministic CPU reference.

## 5. Outputs
- Median GPU time by temporary-count variant.
- Slowdown relative to the lightest register-pressure path.
- Evidence of sensitivity to the live-state schedule; compiler/profiler measurements are needed to establish registers, spills, or occupancy.

## 6. Interpretation
- This is a proxy experiment, so it should be read as pressure sensitivity rather than a literal register count measurement.
- Source batch width does not prove allocated register count. Loop scheduling, compiler optimization, and any spills remain possible causes of timing changes.
