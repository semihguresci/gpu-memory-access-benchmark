# Experiment 15: Bandwidth Saturation Sweep

## 1. Lecture Focus
- Concept: Scaling otherwise identical contiguous workloads until achieved memory throughput reaches a sustained practical plateau.
- Why this matters: Later optimization experiments only make sense when run sizes are large enough that fixed dispatch overhead is already amortized.
- Central question: How large must the workload be before achieved GB/s stabilizes, and how stable is that plateau across simple memory modes?

## 2. Learning Objectives
By the end of this experiment, you should be able to:
- explain the performance mechanism behind the studied concept
- design a controlled Vulkan compute benchmark for this concept
- interpret measured results without over-claiming causality
- document practical rules you would apply in production kernels

## 3. Theory Primer (Lecture Notes)
- Distinguish overhead-bound behavior from steady-state bandwidth-bound behavior: small dispatches often under-report the device's practical throughput because launch and setup costs dominate.
- Use simple contiguous kernels so transaction shape stays near the "good path" and problem size becomes the primary changing variable.
- Treat the plateau as sustained measured bandwidth on the tested hardware, not as proof of theoretical peak or full-memory-system utilization.
- Keep upload, readback, and correctness work separate from the dispatch timing metric so the reported GB/s reflects kernel execution rather than host-device staging.

## 4. Hypothesis
All three simple contiguous modes should climb quickly from an overhead-bound region into a flatter sustained-throughput region; `read_write_copy` should provide the most representative plateau for later experiments, while `read_only` may level out differently because of caching behavior.

## 5. Experimental Design
### Independent variables
Memory modes: `read_only`, `write_only`, `read_write_copy`.

Dense size sweep: `1 MiB`, `2 MiB`, `4 MiB`, `8 MiB`, `16 MiB`, `24 MiB`, `32 MiB`, `48 MiB`, `64 MiB`, `96 MiB`,
`128 MiB`, `192 MiB`, `256 MiB`, `384 MiB`, `512 MiB`, `768 MiB`, `1 GiB` (runtime-clamped when the device or
`--size` budget cannot support the full set).

### Controlled variables
Same contiguous 1D indexing, same scalar type (`float`), same workgroup size, same dispatch count, same validation
policy, and the same resident-buffer model for every measured point.

### Workload design
Reuse the same simple contiguous memory behavior from Experiment 03, but with a denser size sweep and Experiment 15
shader basenames so the result is a focused saturation study instead of a broad baseline comparison.

## 6. Implementation Plan
1. Implement the three simple contiguous kernels under `shaders/15_bandwidth_saturation_sweep/` with unique `15_` basenames.
2. Generate the dense size sweep on the host and clamp it against both `--size` and device dispatch limits.
3. Keep deterministic correctness validation for unchanged source contents, deterministic write outputs, and copy outputs.
4. Run warmup iterations before measured iterations.
5. Capture raw timing plus size-specific metadata for every run.
6. Export machine-readable results for plateau analysis and chart generation.

## 7. Measurement Protocol
- Timing source: GPU timestamp queries for dispatch timing.
- Reporting: median as primary, p95 as stability indicator, plus a plateau-onset estimate derived during analysis.
- Run policy: multiple repetitions per point, deterministic initialization patterns, and one dispatch per timed sample.
- Metadata: GPU model, driver, Vulkan version, OS, compiler flags, shader mode, and realized size sweep.

## 8. Data to Capture
GB/s versus size, plateau onset, sustained-region median GB/s, and variability at larger sizes.

Recommended columns:
- experiment_id
- variant
- problem_size
- iteration
- gpu_ms
- throughput
- gbps
- correctness_pass
- notes

## 9. Expected Patterns and Interpretation

Small points should be visibly overhead-bound, mid-sized points should ramp toward a knee, and larger points should
flatten into a device-specific steady region rather than improving indefinitely.

Interpretation checklist:
- confirm correctness before comparing performance
- separate the ramp-up region from the sustained-throughput plateau
- compare the resulting plateau against Experiment 03 mode baselines and use it to choose representative sizes for Experiments 16-20
- treat the plateau as practical measured evidence on one device, not as proof of spec-sheet peak bandwidth

## 10. Common Failure Modes
Including upload/readback time in the dispatch metric, declaring saturation from too few large-size points, and letting thermal or boost drift distort the late-run plateau.

## 11. Deliverables
Saturation curve with plateau marker(s), a plateau summary table, and practical workload-size guidance.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use the measured plateau-region sizes as the default operating range for shared-memory, tile-size, register-pressure,
and synchronization studies in Experiments 16-20.


