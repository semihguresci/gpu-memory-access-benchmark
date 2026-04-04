# Experiment 14: Read Reuse and Cache Locality

## 1. Lecture Focus
- Concept: Temporal locality and controlled reuse-distance effects.
- Why this matters: Many production kernels reread hot inputs, and the value of a good access schedule depends on whether reuse arrives before data ages out of effective cache.
- Central question: How rapidly does the benefit of a second read decay as the gap between touches widens?

## 2. Learning Objectives
By the end of this experiment, you should be able to:
- explain the performance mechanism behind the studied concept
- design a controlled Vulkan compute benchmark for this concept
- interpret measured results without over-claiming causality
- document practical rules you would apply in production kernels

## 3. Theory Primer (Lecture Notes)
- Distinguish spatial locality from temporal locality: even when first touches stay contiguous, the timing of the second touch can still change performance.
- Treat reuse distance as "how many other reads happen before the same element is touched again".
- Expect benefit only while the reused footprint fits within an effective cache residency window; beyond that, the second touch behaves closer to a fresh read.
- Use a design that avoids mistaking compiler register reuse or loop-invariant elimination for memory-system behavior.

## 4. Hypothesis
Paired rereads with short reuse distance outperform far-distance paired rereads; a full-span second pass approaches the no-useful-reuse baseline.

## 5. Experimental Design
### Independent variables
Reuse schedules: `reuse_distance_1`, `reuse_distance_32`, `reuse_distance_256`, `reuse_distance_4096`, `reuse_distance_full_span`.

### Controlled variables
Same logical invocation count, same two touches per unique source element, same sequential destination write pattern, same arithmetic, same workgroup size, and same source element type.

### Workload design
Host-generated index schedules where each source element appears exactly twice and only the gap between those two appearances changes.

## 6. Implementation Plan
1. Use a gather-style shader with one source read and one sequential destination write per invocation.
2. Generate deterministic pair-reuse index schedules on the host and keep the source buffer fixed to `count / 2` unique elements.
3. Add exact CPU reference validation for destination values and index-buffer invariance.
4. Run warmup iterations before measured iterations.
5. Capture raw timing, reuse-distance metadata, and environment details for every run.
6. Export machine-readable results for plotting slowdown and speedup curves.

## 7. Measurement Protocol
- Timing source: GPU timestamp queries for dispatch timing.
- Reporting: median as primary, p95 as stability indicator, plus normalized speedup versus `reuse_distance_full_span`.
- Run policy: multiple repetitions per point, even logical count, deterministic schedule generation.
- Metadata: GPU model, driver, Vulkan version, OS, compiler flags, and reuse-schedule parameters.

## 8. Data to Capture
Runtime, reuse-distance sensitivity, speedup versus the far-distance baseline, and stability across reuse distances.

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

Short-distance pairs should produce the best times; benefit should decay with larger reuse blocks and flatten between the largest-distance and full-span cases.

Interpretation checklist:
- confirm correctness before comparing performance
- separate the cache-helped region from the long-distance plateau
- compare against Experiment 12 distributions and Experiment 15 saturation data where relevant
- treat any plateau as architecture-specific evidence, not as a direct proof of cache size

## 10. Common Failure Modes
Accidentally benchmarking compiler/register reuse instead of memory reuse, incorrect tail handling for partial pair blocks, and silently changing unique-source count across variants.

## 11. Deliverables
Reuse-distance curve, normalized speedup table, and a short locality interpretation.

Minimum artifact set:
- one results table (csv/json)
- one chart showing the primary sweep
- one short analysis section with explicit limitations

## 12. Follow-Up Link
Use the measured locality window to justify tiling, staging, and ordering choices in Experiments 16-17 and later rendering-adjacent systems.


