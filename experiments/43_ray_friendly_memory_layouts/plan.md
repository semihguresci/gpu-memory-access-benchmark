# Advanced Investigation 09: Memory System Study for Ray-Friendly Layouts

## 1. Lecture Focus
- Concept: Traversal-friendly layout behavior under coherent/incoherent query patterns.
- Why this matters: Builds strong foundation for future ray traversal systems without requiring full ray tracer implementation.
- Central question: Which layouts maintain best locality and throughput under varying query coherence?

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
Layouts optimized for coherent traversal degrade less under moderate incoherence and benefit from ordering.

## 5. Experimental Design
### Independent variables
Layout variant, query order pattern, coherence level, reorder strategy.

### Controlled variables
- Fixed benchmark harness and timing method (GPU timestamp queries).
- Fixed data generation seeds per scenario where reproducibility is needed.
- Fixed correctness oracle per variant.

### Metrics
Bytes fetched, throughput, locality sensitivity, reorder gain.

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
Coherence comparison chart, layout ranking, traversal-oriented guidelines.

Minimum artifact set:
- one core chart
- one summary table
- one short conclusions page with limitations

## 9. Portfolio Framing Notes
- Frame conclusions as measured observations plus reasoned interpretation.
- Avoid claiming universal behavior from one GPU unless cross-GPU validated.
- Highlight tradeoffs and failure modes, not just best numbers.
