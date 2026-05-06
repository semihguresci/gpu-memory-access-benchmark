# Advanced Investigation 02: BVH Node Layout Experiments

## 1. Lecture Focus
- Concept: Node representation impact on traversal efficiency.
- Why this matters: Directly relevant to ray traversal, visibility, collision, and spatial querying pipelines.
- Central question: How do AoS, SoA, wide-node, and compressed-node layouts affect traversal behavior?

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
Layout decisions that reduce bytes/query and improve coherence lower traversal cost even with decode overhead.

## 5. Experimental Design
### Independent variables
Node layout type, node width, metadata compression level, scene distribution.

### Controlled variables
- Fixed benchmark harness and timing method (GPU timestamp queries).
- Fixed data generation seeds per scenario where reproducibility is needed.
- Fixed correctness oracle per variant.

### Metrics
Traversal time, nodes visited, bytes/query, coherence sensitivity.

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
Layout comparison matrix, memory-footprint chart, traversal recommendations.

Minimum artifact set:
- one core chart
- one summary table
- one short conclusions page with limitations

## 9. Portfolio Framing Notes
- Frame conclusions as measured observations plus reasoned interpretation.
- Avoid claiming universal behavior from one GPU unless cross-GPU validated.
- Highlight tradeoffs and failure modes, not just best numbers.
