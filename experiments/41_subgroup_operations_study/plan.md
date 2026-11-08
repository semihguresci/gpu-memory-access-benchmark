# Advanced Investigation 06: Subgroup Operations Study

## 1. Lecture Focus
- Concept: Warp/wave-level intrinsics versus shared-memory patterns.
- Why this matters: Subgroup features can remove barriers and temporary memory overhead in key primitives.
- Central question: Where do subgroup operations outperform shared-memory implementations?

## 2. Learning Objectives
By the end of this investigation, you should be able to:
- justify why this systems-level problem matters in practical GPU pipelines
- design a controlled benchmark matrix with clear independent variables
- interpret results without confusing correlation and causation
- extract design rules and limitations suitable for portfolio presentation

## 3. Theory Primer (Lecture Notes)
- Start with a pipeline-level mental model, not just a kernel-level view.
- Identify resource bottlenecks: memory traffic, synchronization, occupancy pressure, and control-flow efficiency.
- Separate algorithmic cost from implementation artifacts.
- Record assumptions and known unknowns before running the benchmarks.

## 4. Hypothesis
Subgroup-assisted workgroup aggregation can reduce explicit shared-memory coordination compared with a shared tree, including the cost of combining subgroup partials.

## 5. Experimental Design
### Independent variables
Shared-tree versus subgroup-assisted workgroup aggregate and data size; subgroup size/support are observed device properties.

### Controlled variables
- Fixed benchmark harness and timing method (GPU timestamp queries).
- Fixed data generation seeds per scenario where reproducibility is needed.
- Fixed correctness oracle per variant.
- Both variants return the same workgroup-wide `uint32` sum plus `17 * odd_input_count` to every invocation. Zero-pad out-of-range invocations. No local-ID/subgroup-lane mapping is assumed.
- The shared path reduces 256 local-ID-indexed inputs; the subgroup path writes one sum and ballot count per actual subgroup, then combines `gl_NumSubgroups` partials. Both use at most 2 KiB shared memory.
- Compare every output to an independent CPU workgroup aggregate after every timed iteration; initialize each output to the complement of its expected result before dispatch.

### Metrics
GPU time, logical elements/s and requested payload GB/s (one 4-byte input load and one 4-byte output store per element). This is a full workgroup primitive, not timing of a subgroup instruction in isolation or observed DRAM bandwidth.

## 6. Implementation Plan
1. Implement minimally correct baseline variant first.
2. Add one optimized variant at a time to preserve causal clarity.
3. Add deterministic correctness tests and edge-case datasets.
4. Run warmup plus repeated measured runs for each matrix point.
5. Export raw data and metadata to versioned result files.
6. Generate charts and write a short interpretation section with caveats.

## 7. Analysis Prompts
- Which stage or operation dominates total cost and why?
- Which tuning parameter is most sensitive?
- Which findings are likely architecture-dependent?
- What would change in a production rendering/compute pipeline?

## 8. Deliverables
Subgroup vs shared-memory charts, portability notes, implementation policy.

Minimum artifact set:
- one core chart
- one summary table
- one short conclusions page with limitations

## 9. Portfolio Framing Notes
- Frame conclusions as measured observations plus reasoned interpretation.
- Avoid claiming universal behavior from one GPU unless cross-GPU validated.
- Highlight tradeoffs and failure modes, not just best numbers.
