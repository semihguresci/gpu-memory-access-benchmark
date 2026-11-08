# Advanced Investigation 13: Dependent Load Latency and Memory-Level Parallelism

## 1. Lecture Focus
- Concept: Separating per-load latency from memory-level parallelism (MLP).
- Why this matters: A strided or hashed access pattern loses performance either because each dependent load is slow, or because too few loads are outstanding. These have different fixes: placement/reuse vs. batching, unrolling, and more independent work per invocation.
- Central question: How does the effective cost of one dependent load change as each invocation is given 1, 2, 4, 8, or 16 independent load chains of shortened length at constant total load count?

## 2. Learning Objectives
By the end of this investigation, you should be able to:
- explain why pointer-chasing isolates per-load latency on a GPU
- interpret load counts per second as an effective-latency metric, without claiming it is an isolated hardware latency
- describe the interaction between chain count, occupancy, and cache level
- report layout conclusions that distinguish latency-bounded from bandwidth-bounded kernels

## 3. Theory Primer (Lecture Notes)
- A dependent load cannot issue until the previous load's address is available, so pointer chasing serializes requests.
- More independent chains per invocation allow the memory system to overlap misses across chains; the observed per-load cost drops toward the saturation regime known from Experiment 15.
- A random permutation visits each index once per full cycle; short chains may re-enter short cycles, which is inherent to pointer chasing and documented as a limitation.
- `ns per dependent load` is an effective aggregate cost, including queue occupancy effects. It needs profiler support to be decomposed into pill latency, address translation, and scheduling.

## 4. Hypothesis
At small working sets (cache-resident) the single-chain case approaches core-local latency; with 8–16 chains the observed per-load cost drops by an architecture-dependent factor that this experiment measures directly.

## 5. Experimental Design
### Independent variables
- Independent chain count per invocation: 1, 2, 4, 8, 16 (per-chain walk steps shortens as 64 / chain_count so total dependent loads stay 64).
- Each walk step advances every active chain before the next step. Chain count is a pipeline specialization constant, and each chain has a separate scalar address, exposing independent loads without dynamically indexed private storage.
- Pointer-table problem size: 64 Ki to 2 Mi entries at the default 32 MiB total scratch budget, extended to the amount supported when larger scratch sizes are requested.

### Controlled variables
- Total requested loads per invocation (64) and output size (one 4-byte result per invocation) are constant across chain variants at a given problem size.
- One randomized Fisher–Yates permutation table with a fixed seed is uploaded once per problem size into device-local memory; every variant reads the same table.
- All non-benchmark transfers (sentinel refill and readback copies) run outside the timed dispatch, and their cost is identical for every variant.
- Workgroup size (256), dispatch shape, and push-constant layout remain constant. Chain count uses separate specialized pipelines; the available parallelism and live address-register count change together, so register pressure and resulting occupancy remain limitations rather than controlled quantities.

### Metrics
- Dispatch GPU ms per case; derived aggregate `loads per second` = problem_size × 64 / GPU seconds.
- `ns per dependent load` = GPU ms × 1e6 / (problem_size × 64); this is an effective cost, not an isolated hardware latency.
- Stability via p95 / median; per-iteration FNV-64 checksum for correctness.

## 6. Implementation Plan
1. Generate a randomized permutation table on the CPU with a fixed seed.
2. Stage the table into device-local memory once per problem size.
3. Specialize the chain-walking shader once per chain count. Its bounds-guarded step loop loads every active scalar chain address before starting the next step; inactive chain states and branches are compile-time eliminated. Push constants carry the element count and chain-start stride.
4. Compute the CPU oracle for every chain count and problem size, reusing full-array gathered walks.
5. Validate per iteration against that oracle and record an FNV-64 checksum.
6. Export rows with the chain configuration, interleaved issue order, specialization/scalar-state implementation, and a basis note for each metric formula.

## 7. Analysis Prompts
- How much does the per-chain time ratio change between chain counts 1 and 16 at each working set size?
- Which working set boundary (or cache level) shifts chain scaling?
- What does this predict about the loop-level performance of pointer-heavy versus batched gather kernels in Experiments 12, 27, and 45?

## 8. Deliverables
Per-load latency chart by problem size, chain-scaling ratio table, stability overview, limitations section describing cycle re-entry and aggregate nature of the latency estimate.

Minimum artifact set:
- one core chart (`dependent_load_latency_ns_per_load_vs_size.png`)
- one summary table (`dependent_load_latency_summary.csv`) plus latency table
- one short conclusions page with limitations (`results.md`)

## 9. Portfolio Framing Notes
- Frame as: "I can measure whether a memory-bounded kernel is latency-saturated or MLP-saturated, and what chain batching is worth."
- Avoid claiming an isolated hardware latency from timing; profiler work would be required to reach that claim.
- The single-sample GPU/driver and working-set limits are explicit scope boundaries in every report.
