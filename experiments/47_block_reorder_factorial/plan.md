# Advanced Investigation 15: Coalescing, Spatial Locality, and Cache Residency Factorial

## 1. Lecture Focus
- Concept: A 2x2 factorial varying subgroup address density and block visit order, with an explicit warm-versus-evicted residency control. Timing ratios describe these factors; transaction counters are required to attribute a slowdown to physical transactions.
- Why this matters: Real kernels change lane mapping, visit order, and cache temperature at the same time. Current throughput rankings in Experiments 11, 12, 14, 27, and 45 combine these effects; this factorial holds the visited dataset bijective and constant so each factor's contribution is measured separately.
- Central question: How much slowdown comes from permuted lane mapping inside aligned blocks, how much from shuffled block visit order, and how do warm and evicted cache states change each answer?

## 2. Learning Objectives
By the end of this investigation, you should be able to:
- design a factorial access-pattern matrix with an exactly bijective visit contract
- explain why reversal across blocks larger than a subgroup spreads each subgroup's source requests without changing the dispatch's bijective visited set
- read factor effect ratios and their additivity instead of single "bad pattern" summaries
- interpret warm-versus-evicted comparisons as a reproducible perturbation, not a certified cold cache

## 3. Theory Primer (Lecture Notes)
- A permutation confined to one subgroup leaves its address set unchanged and therefore does not provide a coalescing control. Each block here spans 2, 4, 8, 16, or 32 reported subgroup widths, so reversal spreads a subgroup's requests across a larger region while preserving the dispatch's visited set.
- Block shuffling preserves block-internal density but destroys stream prefetch and page-level locality.
- Logical input positions follow dense active subgroup/lane order, constructed with ballot ranks and a workgroup prefix. This keeps the oracle correct when Vulkan assigns local IDs irregularly or launches partial subgroups. All variants execute this same setup, use the same reversal arithmetic and mask selection, and request an identity or shuffled block-table entry per invocation.
- On the 32-lane reference GPU, block widths are 64, 128, 256, 512, and 1024 words. With full aligned subgroups, ordered requests touch four 32-byte sectors and reversed requests touch 8, 16, 32, 32, and 32 sectors respectively. This address calculation is a prediction, not a measurement of cache or DRAM transactions.
- Cache residency is torn between two experiments: a warm steady state and an evicted state produced by streaming a dedicated 64 MiB device-local buffer (roughly 25x the reference device's L2) once outside the timed dispatch. The sweep warms the eviction buffer's own tail and leaves hardware prefetch state behind, so it is a reproducible perturbation whose residual direction is documented, not a certified cold cache; the source dataset itself is never touched by the sweep.
- The 64 MiB eviction buffer counts toward the total `--size` payload budget. The remaining budget is split across source, output, block-permutation, upload-staging, and readback-staging buffers. Use `--size 128M` by default; budgets of 64 MiB or less fail before allocation. Every row reports all six live buffers, their payload and allocation totals, and the eviction span, including warm rows.
- Before every measured dispatch, warm cases execute an unmeasured priming dispatch and evicted cases execute the dedicated sweep. Both then upload the complement of every expected output word. A skipped write therefore fails validation even when a prior variant produced the same result. The common output copy is outside dispatch timing and does not access the input, but can perturb cache state; warm/evicted labels describe this reproducible setup rather than certified residency.

## 4. Hypothesis
Reversal can increase request dispersion until each active lane requests a different sector. Block shuffling can change cache, translation, and stream behavior at fixed subgroup density. Neither slowdown nor eviction response is assumed monotonic; all effects are measured at matched widths and problem sizes.

## 5. Experimental Design
### Independent variables
- Block width: reported `subgroupSize` multiplied by 2, 4, 8, 16, or 32. Every problem size is a multiple of both the 256-invocation workgroup size and the largest block width.
- Block visit order: sequential vs seeded shuffled block permutation.
- Lane mapping: ordered vs bit-reversal within the block.
- Residency: warm steady state vs evicted (a dedicated 64 MiB device-local buffer streamed outside the timed region); evicted variants run at the smallest and largest selected block widths.

### Controlled variables
- Every source element is visited exactly once per dispatch (bijective visit contract with no replacement).
- One 4-byte source read, one 4-byte block-table read, the same transform, and one 4-byte result write per invocation; output addresses are identical across variants. Sequential variants use an identity block table and shuffled variants use a seeded permutation.
- One fixed-seed source pattern (elementwise hash values) staged to device-local memory per size; identity or fixed-seed block permutation staged before each variant, outside dispatch timing.
- Workgroup size (256), dispatch shape, push-constant layout, and per-variant index-buffer spans are exported, not hidden.
- Requires compute-stage basic and ballot subgroup operations, a reported power-of-two subgroup size in [2, 256], and 2048 bytes of shared memory for dense membership mapping. Storage ranges and dispatch counts are capped against device limits.

### Metrics
- Dispatch GPU ms per case; GB/s over `estimated_shader_request_bytes = element_count * 12` for every variant. The model counts each invocation's block-table request separately even when requests share an address or are broadcast/cached. The unique block-table span remains separate metadata; the byte model does not estimate physical DRAM traffic.
- Factor effect ratios versus the same-width sequential/ordered baseline; evicted-over-warm residency ratios at matched sizes.
- Stability via p95/median and per-iteration FNV-64 checksums for the exact-output oracle.

## 6. Implementation Plan
1. Stage the source dataset once per problem size into device-local memory.
2. Generate identity and seeded shuffled tables per (size, width), and stage the selected table before each variant.
3. Dispatch the shared main shader per access variant with push-constant modes.
4. For evicted variants, dispatch the sweep shader before the timed dispatch, on its own submission. Launch at most one invocation per discard-output word; each invocation reduces its disjoint strided chunk sequence into that word. Every eviction-buffer word is read exactly once, each discard word has one writer, and sweep failures fail the case.
5. Validate every timed iteration against the CPU oracle and record checksums.
6. Export `estimated_request_bytes` per variant and document every metric basis in row notes.

## 7. Analysis Prompts
- Do within-reversal and block-shuffle slowdowns multiply (additivity holds)?
- At which working set does block shuffling start to hurt, and does that threshold track cache capacity?
- Does eviction change any ranking, or only absolute spread? Use this to decide whether resideness explains Experiment 27's random-vs-sequential gap.

## 8. Deliverables
Factor effect ratio tables, residency ratio tables, ms-by-size overview, stability overview, and a limitations section covering bijectivity assumptions, bit-reversal specifics, eviction perturbation, and the absence of transaction counters.

Minimum artifact set:
- one core chart (effect ratios by block width)
- one summary table plus effect and residency tables
- one short conclusions page with limitations (`results.md`)

## 9. Portfolio Framing Notes
- Frame as: "I can measure matched address-dispersion, block-order, and residency factors while keeping coverage, requested bytes, and indexing work constant."
- Avoid claiming exact transaction counts without profiler counters; the ratio tables show observed time factors only.
- Note that evicted residency is a documented perturbation; cache-capacity claims require counter or tooling support.
- Record that the smallest problem sizes produce single-digit-microsecond dispatches where timestamp quantization inflates ratio noise; effect ratios are strongest at the largest working sets.
- Note that warm-versus-evicted comparisons share a clock-state confound: the sweep keeps the GPU busier, so it measures cache state together with DVFS behavior, not either in isolation.
- Dense subgroup mapping and common reversal arithmetic are included in dispatch timing. Driver instruction selection, partial subgroup membership, and physical transaction behavior require profiler inspection before stronger causal claims.
