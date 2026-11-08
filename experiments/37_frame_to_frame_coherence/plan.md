# Advanced Investigation 10: Frame-to-Frame Coherence Studies

## 1. Lecture Focus
- Concept: Temporal stability and ordering reuse effects.
- Why this matters: Real-time systems benefit from stable data ordering and temporal locality.
- Central question: How do deterministic source remappings affect an eight-frame read/transform/write sequence?

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
Stable ordering improves locality and reduces variance across neighboring frames.

The current proxy compares coherent shifts, block-local rotations and hashed remaps. These also change spatial locality and source collisions, so timing cannot isolate temporal locality or establish a rebuild/traversal effect.

## 5. Experimental Design
### Independent variables
Source-index remapping mode and logical source size.

### Controlled variables
- Fixed benchmark harness and timing method (GPU timestamp queries).
- Fixed data generation seeds per scenario where reproducibility is needed.
- Fixed correctness oracle per variant.
- Eight frames each execute one source load, the same integer transform and one output store per logical element. Seven compute barriers order the frame sequence.
- Preserve all eight disjoint frame outputs and validate every value against the CPU oracle after each timed iteration. Initialize outputs to the complement of their expected values to expose missed writes.
- Budget one source span plus eight output spans within total CLI scratch; distinguish buffer payload, allocation padding and active spans in row notes.

### Metrics
Sequence GPU time, processed frame-elements/s and shader payload GB/s (`N * 8 * 8` bytes). Requested bytes do not establish DRAM traffic. Host output initialization and full validation are outside GPU timing.

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
Stable vs unstable charts, temporal-locality notes, pipeline recommendations.

Minimum artifact set:
- one core chart
- one summary table
- one short conclusions page with limitations

## 9. Portfolio Framing Notes
- Frame conclusions as measured observations plus reasoned interpretation.
- Avoid claiming universal behavior from one GPU unless cross-GPU validated.
- Highlight tradeoffs and failure modes, not just best numbers.
