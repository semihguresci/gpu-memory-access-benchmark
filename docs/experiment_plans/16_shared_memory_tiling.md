# Experiment 16: Shared or Workgroup Memory Tiling

## 1. Lecture Focus
- Concept: Staging overlapping neighborhoods into on-chip workgroup memory so threads can reuse data instead of re-reading global memory.
- Why this matters: Shared-memory tiling is a foundational GPU optimization, but it trades lower global-memory traffic for halo loads, barrier cost, and local-memory pressure.
- Central question: For a fixed stencil-like workload, when does explicit workgroup-memory tiling outperform direct global reads?

## 2. Learning Objectives
By the end of this experiment, you should be able to:
- explain the performance mechanism behind the studied concept
- design a controlled Vulkan compute benchmark for this concept
- interpret measured results without over-claiming causality
- document practical rules you would apply in production kernels

## 3. Theory Primer (Lecture Notes)
- Start from the execution model: workgroups, waves/warps, and memory transactions.
- Identify whether the kernel is likely memory-bound, latency-bound, or synchronization-bound.
- Predict how this experiment changes transaction shape, locality, pressure, or control-flow efficiency.
- Record assumptions explicitly before measuring so conclusions can be tested, not guessed.

## 4. Hypothesis
For a fixed workgroup size, a tiled kernel outperforms a direct global-read kernel once neighborhood reuse is large enough that fewer global loads outweigh halo-loading and barrier overhead.

## 5. Experimental Design
### Independent variables
- Implementation variant: `direct_global` vs `shared_tiled`
- Reuse radius: `1`, `4`, `8`, `16`
- Logical output count: one large steady-state size derived from `--size`

### Controlled variables
- Same 1D stencil arithmetic and deterministic source data
- Same default `local_size_x` and same logical output count across variants
- Same output type, correctness rules, warmup policy, and timing path

### Workload design
- Use a 1D sliding-window sum over a padded source buffer.
- Each invocation writes one output element using `2 * radius + 1` source values.
- The direct variant reads the full neighborhood from device memory for every output.
- The tiled variant cooperatively loads `local_size_x + 2 * radius` elements into workgroup memory, executes one barrier, and computes the same output from staged values.
- Keep tile size fixed in Experiment 16 so Experiment 17 can isolate tile-size tuning.

## 6. Implementation Plan
1. Implement or select the shader variant(s) that isolate this concept.
2. Wire host-side sweep parameters into CLI/config.
3. Add correctness checks against deterministic CPU reference outputs.
4. Run warmup iterations before measured iterations.
5. Capture raw timing and metadata for every run.
6. Export results in machine-readable format for plotting.

## 7. Measurement Protocol
- Timing source: GPU timestamp queries for dispatch timing.
- Reporting: median as primary, p95 as stability indicator.
- Run policy: multiple repetitions per point, fixed seeds for reproducibility.
- Metadata: GPU model, driver, Vulkan version, OS, compiler flags, shader options.

## 8. Data to Capture
Runtime, speedup factor, estimated global-traffic reduction, barrier count, and shared-memory footprint.

Recommended columns:
- experiment_id
- variant
- problem_size
- iteration
- gpu_ms
- throughput
- gbps
- correctness_pass
- notes (`reuse_radius`, `local_size_x`, `group_count_x`, `tile_span_elements`, `shared_bytes_per_workgroup`,
  `barriers_per_workgroup`, `estimated_global_read_bytes`)

## 9. Expected Patterns and Interpretation

Radius `1` may show little or no gain, while larger radii should increasingly favor the tiled path if the kernel is traffic-bound.

Interpretation checklist:
- confirm correctness before comparing performance
- separate reuse-driven speedup from barrier/local-memory overhead
- compare against baseline experiments (01, 03, 04, 15 where relevant)
- do not generalize tile-size conclusions yet; that belongs to Experiment 17
- highlight both absolute and normalized deltas

## 10. Common Failure Modes
Incorrect halo loading, missing barriers, tile-tail indexing bugs, or over-claiming cache-versus-shared-memory causality from timing alone.

## 11. Deliverables
Direct-vs-tiled speedup chart by reuse radius and one short barrier-cost commentary.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Follow with the tile-size occupancy tradeoff study in Experiment 17 using the same tiled kernel shape.


