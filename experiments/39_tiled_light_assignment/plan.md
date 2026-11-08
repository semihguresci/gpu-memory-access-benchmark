# Advanced Investigation 04: Tiled Light Assignment

## 1. Lecture Focus
- Concept: Light-to-tile overlap counting strategies.
- Why this matters: Highly relevant to Forward+, tiled deferred, and many-light rendering techniques.
- Central question: How do append/list-build strategies trade contention, load balance, and throughput?

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
Light-centric atomic counters and tile-centric serial/shared reductions trade atomic contention against repeated light scans. The current proxy builds counts, not assignment lists or prefix offsets.

## 5. Experimental Design
### Independent variables
Light count, uniform/clustered light distribution and light-centric atomic versus tile-centric serial/shared-reduction strategy at fixed tile dimensions.

### Controlled variables
- Fixed benchmark harness and timing method (GPU timestamp queries).
- Fixed data generation seeds per scenario where reproducibility is needed.
- Fixed correctness oracle per variant.
- Validate every tile count against the deterministic CPU overlap oracle after each timed iteration. Derive total overlaps from the measured tile counts on the CPU outside GPU timing; no global statistics atomics run in any strategy.

### Metrics
GPU time and logical lights/s. Export modeled requested global bytes: light-centric loads each light once and models one useful tile atomic as a 4-byte read plus 4-byte write per overlap; tile-centric paths read all lights per tile and store one count per tile. Include the common 4-byte mode echo. These estimates are not observed cache/DRAM traffic, and the algorithms intentionally perform different candidate tests.

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
Light count scaling chart, tile-size map, contention discussion.

Minimum artifact set:
- one core chart
- one summary table
- one short conclusions page with limitations

## 9. Portfolio Framing Notes
- Frame conclusions as measured observations plus reasoned interpretation.
- Avoid claiming universal behavior from one GPU unless cross-GPU validated.
- Highlight tradeoffs and failure modes, not just best numbers.
