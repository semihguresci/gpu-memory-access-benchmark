# Experiment review and memory research expansion plan

Review date: 2026-10-23. Scope: all 45 enabled experiments, their plans/reports, C++ implementations/adapters, shaders, analysis scripts, and shared measurement/export/build code.

This document preserves the **pre-fix review snapshot**. The fixes and verification are recorded in [Experiment integrity fixes](experiment_integrity_fixes.md); current per-experiment reports supersede the historical measurements quoted below.

The suite has broad coverage of throughput, data layouts, coalescing, workgroup memory, and parallel primitives. The largest missing dimensions are dependent-load latency, memory-level parallelism, controlled cache residency, image access, transfer amortization, and actual concurrent queues. Fix the correctness and accounting defects below before using new collections to support architectural conclusions.

## Evidence and verification

- Configured with `cmake --preset windows-tests-vs`; built the Release benchmark and both test targets with shader compilation enabled. Build succeeded.
- `ctest --test-dir build-tests-vs -C Release --output-on-failure`: **40/40 passed**, including the existing ten-experiment GPU smoke matrix. This does not rerun every experiment or prove portability/causal isolation.
- Checked all **423 links** in the 45 results reports: every local target exists in this workspace. All **6,485 current exported rows** say correctness passed; the oracle limitations below still apply.
- Reports **01–23** link an archived run different from their linked current `benchmark_results.json`. Reports **24–45** link archives equal to their current raw exports. For 33–45, independently recomputed summary medians match raw rows within CSV rounding.
- Reproduced the export failure in finding 14. Also ran a small Experiment 01 case with validation enabled and an aligned `--size 3M`; no synchronization-hazard messages were emitted. The source-level dependency defect below remains; this run does not establish comprehensive synchronization coverage.
- Review-time device: RTX 2080 SUPER, NVIDIA driver 610.88, Vulkan device API 1.4.341. Existing April benchmark artifacts use an older driver/API; review-time smoke data is not a replacement performance collection.

The original review was read-only. Implementation changes and refreshed artifacts belong to the subsequent remediation described in the linked fixes report.

## Findings to address before collecting new results

Priorities: **P1** correctness/invalid access; **P2** material benchmark, portability, accounting, or provenance defect; **P3** narrower metadata/report issue. Line numbers refer to the source snapshot reviewed.

### 1. P1 — Experiment 01 has an unguarded tail write

[Write shader](../shaders/01_dispatch_basics/01_dispatch_basics_write.comp), lines 14–16, writes every launched invocation without checking the logical count. [Count selection](../src/experiments/dispatch_basics_experiment.cpp), lines 47–60, includes the exact memory-limited count, while dispatch rounds up to 64 lanes. The default 4 MiB total scratch budget is divided among three buffers: 349,525 floats cause 349,568 invocations, including **43 out-of-range writes**. In-range readback cannot detect these stores. Pass a logical count and guard the tail, or choose a correctly aligned count.

### 2. P1 — Experiment 20 overwrites shared state before all consumers finish

[Flat shader](../shaders/20_barrier_synchronization_cost/20_barrier_synchronization_cost_flat.comp), lines 69–73, and [tiled shader](../shaders/20_barrier_synchronization_cost/20_barrier_synchronization_cost_tiled.comp), lines 73–77, synchronize the producer write before reading `shared_seed`, but omit a consumer-completion barrier before the next overwrite. One subgroup can advance while another still needs the old seed. Add the missing dependency or use separate phase storage, then update the measured barrier count. Also hold arithmetic and shared communication constant when studying barrier frequency; current variants change those too.

### 3. P1/P2 — Subgroup implementations assume unsupported invocation ordering

[31 scan](../shaders/31_subgroup_scan_variants/31_subgroup_scan_subgroup.comp), lines 30–69, and [32 stable compaction](../shaders/32_subgroup_stream_compaction_variants/32_subgroup_compaction_subgroup_ballot.comp), lines 52–83, derive data positions from local IDs but derive prefixes from subgroup/lane order. [41 software reduction](../shaders/41_subgroup_operations_study/41_subgroup_operations_study.comp), lines 39–58, additionally reads shared partners using `local_index + stride` and roots using subgroup IDs; a different legal mapping can access incorrect elements or exceed the scratch array. Its CPU reference repeats the assumption.

Vulkan explicitly gives no direct relationship between subgroup lane IDs and local invocation IDs. Define a portable logical indexing scheme and matching oracle; where constructing an index from subgroup coordinates, establish full-subgroup constraints and handle support explicitly. See [Khronos Shader Interfaces](https://docs.vulkan.org/spec/latest/chapters/interfaces.html).

### 4. P1/P2 — Repeated writes lack dependencies between dispatches

[01](../src/experiments/dispatch_basics_experiment.cpp), lines 224–229; [04](../src/experiments/sequential_indexing_experiment.cpp), lines 268–272; and [05](../src/experiments/global_id_mapping_variants_experiment.cpp), lines 305–309, repeatedly write the same storage without a compute-to-compute dependency. [02](../src/experiments/local_size_sweep_experiment.cpp), lines 343–348, has the same issue if its public configuration requests more than one dispatch; its current adapter requests one.

A barrier only after the loop does not order the writes inside it. Repeated identical results also cannot establish that every dispatch counted as useful work was observed. Use disjoint output regions or explicit dependencies, and document the resulting timing scope. Vulkan requires memory dependencies for write-after-write hazards: [Khronos synchronization rules](https://docs.vulkan.org/spec/latest/chapters/synchronization.html).

### 5. P2 — Device limits are incompletely enforced

These are related checks with different affected paths:

- **06–10:** counts are constrained by scratch/32-bit sizes without dispatch-limit checks. For example, [06 count selection](../src/experiments/aos_soa_experiment.cpp), lines 609–611, permits 65,536 groups at `--size 1G`, illegal on a device with a 65,535-group limit.
- **17/31:** [17 sizing](../src/experiments/tile_size_sweep_experiment.cpp), lines 155–159, and [31 sizing](../src/experiments/subgroup_scan_variants_experiment.cpp), lines 129–134, size against the largest block although smaller variants need **eight times** as many groups. Bound the common problem size by the smallest legal block or use legal multidimensional/grid-stride dispatch.
- **Storage descriptors:** no `maxStorageBufferRange` check exists repository-wide. Representative [41 sizing](../src/experiments/subgroup_operations_study_experiment.cpp), lines 498–500, and full-range descriptors at lines 250–251 can bind more than a device supports. Validate each representation and auxiliary buffer before allocation/update. See [descriptor range requirements](https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorBufferInfo.html).
- **Shared memory/workgroup size:** 29 declares **33,792 shared bytes**; 35's high variant declares **32,768 bytes**. Their setup does not reject devices below those limits. Many kernels also assume a legal 256-invocation workgroup. Check per-axis size, total invocations, shared footprint, and dispatch dimensions.
- **Subgroup scratch:** 30/31/32 reserve only 32 subgroup entries but accept any nonzero subgroup size. A 256-invocation workgroup with subgroup size 1/2/4 can exceed those arrays. Use actual subgroup-count constraints and sufficient storage, with explicit unsupported-case handling.

### 6. P2 — 06/07 vary field traffic alongside layout

[06 metrics](../src/experiments/aos_soa_experiment.cpp), lines 468–475, use **64 B/particle for AoS versus 48 B for SoA**. Its AoS source writes back unchanged velocity/time fields while SoA writes only the four modified fields. [07 metrics](../src/experiments/aosoa_blocked_layout_experiment.cpp), lines 763–770, use **128 B versus 80 B**, and the AoS source also copies cold fields.

The source programs request different loads/stores, and GB/s uses different numerators for the same useful update. Equalize active field operations; publish common useful-payload bytes separately from estimated/requested traffic. Inspect compiled code before claiming which stores survive optimization. Existing timing is evidence about these implementations, with this confound, rather than an isolated layout effect.

### 7. P2 — 18 changes arithmetic volume with temporary count

[Common shader](../shaders/18_register_pressure_proxy/18_register_pressure_proxy_common.glsl), lines 34–55, iterates over `TEMP_COUNT` during initialization, every fixed-count round, and final folding. `temp_32` executes eight times as many state-update bodies as `temp_4` and computes a different transform. This does not isolate register pressure. Redesign around fixed operations/output with varying live ranges, or describe it as a joint state-size/arithmetic study. Register allocation and spills need compiler/profiler evidence.

### 8. P2 — 39 adds a global observer bottleneck to one strategy

[Light assignment shader](../shaders/39_tiled_light_assignment/39_tiled_light_assignment.comp), lines 104–107, performs both a useful tile-counter atomic and an atomic to the same global statistics word **for every overlap**. Other strategies update statistics only once per tile. This introduces serialization even between disjoint tiles, so the large ranking difference cannot be attributed solely to tile assignment. Derive the overlap total from validated tile counts outside timing, or use an equivalent separate reduction for every strategy, then recollect.

### 9. P2 — 37 validates only one of eight measured frames

[Frame experiment](../src/experiments/frame_to_frame_coherence_experiment.cpp), lines 395–398 and 430–433, constructs/checks only the final frame's reference. All eight frames independently read immutable input and overwrite the same destination; frames 0–6 leave no evidence, although all eight contribute to throughput/bytes. Preserve per-frame outputs or a dependency-aware accumulated oracle, while accounting for its traffic. Separately, current modes change spatial coalescing and collisions too, so temporal reuse is not isolated.

### 10. P2 — 28 exceeds the total scratch budget

[Adapter](../src/experiments/adapters/device_local_vs_host_visible_heap_placement_adapter.cpp), lines 18–19, divides scratch by four. [Allocation](../src/experiments/device_local_vs_host_visible_heap_placement_experiment.cpp), lines 145–197, retains **six equal buffers** simultaneously. `--size 1G` therefore requests 1.5 GiB of buffer payload before allocation padding. Notes at lines 378–379 describe only two/four resident buffers. Reuse staging resources, allocate per active variant, or budget all six; distinguish active working set from all live allocations.

### 11. P2 — Compaction GB/s counts reads that are not executed

[32 accounting](../src/experiments/subgroup_stream_compaction_variants_experiment.cpp), lines 447–449, uses `(N + V + B) * 4`, although input is read only for the `V` survivors. Executed input/output/count payload is `(2V + B) * 4`. In the current 5% case this is **282,907,144 versus 27,894,800 bytes**, a **10.14×** overcount. These are shader payload counts, not physical DRAM bytes.

[24 accounting](../src/experiments/stream_compaction_experiment.cpp), lines 175–184, similarly charges all-`N` source reads despite survivor-only loads, and its staged estimate needs re-derivation from prefix/scatter paths. Keep atomic read-modify-write assumptions explicit. Recompute affected tables and reports; timing/input-elements-per-second are separate metrics.

### 12. P2 — 12 exports a per-buffer budget as total scratch

[Gather notes call](../src/experiments/gather_access_pattern_experiment.cpp), lines 526–528, forwards `config.max_buffer_bytes` into the `scratch_size_bytes` field after the adapter divided the CLI budget by three. Export the original total separately from per-buffer allocation and actual spans. Current metadata understates the requested scratch budget by approximately threefold.

### 13. P2 — Reports and artifacts do not consistently describe the same collection

Every report from **01 through 23** cites an archived collection different from its current raw-export link. Existence checks alone therefore pass while evidence is inconsistent. Examples:

- **06:** report says 10 rows and 1,000,000 particles; current raw has 40 rows and a maximum of 1,048,576.
- **18:** report calls `temp_4` fastest; current raw makes `temp_16` fastest.
- **19/20:** report spreads are about 34.35%/35.96%; current raw spreads are about 0.291%/0.276%.
- **22:** archived/current raw throughput still multiplies elements by three dispatches. The current runtime has already corrected logical throughput to one input stream. Recollect or migrate derived metrics explicitly from raw timing; rerunning an analyzer that trusts old throughput is insufficient.

Publish raw → tables → charts → report as a consistent run bundle, or pin each artifact to immutable run IDs. Record source/binary/shader hashes and analysis version. Historic conclusions should remain labeled as historic, rather than supported by a mutable current-file link.

### 14. P2 — Shared build/export paths can silently invalidate evidence

- [CMake shader command](../CMakeLists.txt), lines 118–123, depends only on the top-level `.comp` file. Editing 18's included common `.glsl` does not trigger shader rebuild. Add compiler-generated include dependencies, using a depfile supported by the chosen CMake generators. A successful C++ build can otherwise execute stale shader code.
- [JSON exporter](../src/utils/json_exporter.cpp), lines 86–87, never checks open/write/flush status, and main prints success. Reproduction with an existing directory named `unwritable_export.json` returned **exit 0**, emitted “Wrote results,” and produced no JSON file. Propagate a recoverable failure or an actionable exception, including write/flush errors.

### 15. P2/P3 — Timing and dispatch metadata need portability fixes

[Timestamp subtraction](../src/utils/gpu_timestamp_timer.cpp), lines 47–52, rejects `end < start` and does not account for the queue's `timestampValidBits`, although [context setup](../src/vulkan_context.cpp), lines 413–422, accepts any nonzero width. Retain the width and handle modulo subtraction for intervals shorter than a wrap, with an explicit maximum interval policy. See [Khronos timestamp queries](https://docs.vulkan.org/spec/latest/chapters/queries.html).

[34 row export](../src/experiments/radix_sort_gpu_experiment.cpp), line 659, says one dispatch although count/scan/scatter execute for every radix pass: 12 or 24 dispatches. Correct that metadata while retaining keys/s based on one logical sort. [45 report](../experiments/45_cross_gpu_reproducibility/results.md), line 33, overstates stride-before-hash ordering across the full sweep; its smallest case reverses those two.

## Coverage of every existing experiment

“Next control” includes deliberate limitations as well as defects; a proxy is not automatically an implementation bug. Most current studies use one GPU/driver and five timed samples.

| ID | What the implementation covers | Most useful next control or limitation |
|---|---|---|
| 01 | Write/no-op, size and repeated dispatch sweeps | Fix bounds/dependencies; validate counted repetitions |
| 02 | Legal local-size sweep, write/no-op | Cross local size with measured resource use at fixed work |
| 03 | Device-local read/write/copy baseline | Warm/evicted buffers; transfer-engine copy; read oracle observes only probes |
| 04 | Contiguous read/add/write | Ordered repetitions; separate streaming from cache-hot replay |
| 05 | Direct, rotated, grid-stride indexing | Equal-launch control; grid-stride changes scheduling too |
| 06 | AoS/SoA particle update | Match field stores; field-touch ratio and placement |
| 07 | AoS/SoA/AoSoA hot/cold fields | Match traffic; block widths and fields touched independently |
| 08 | std140/std430/manual packed record strides | Feature-gated scalar layout and field-access sparsity |
| 09 | vec3 padding, vec4 packing, split scalars | Equal-layout 44/48/64-byte controls; padding and AoS/SoA currently bundled |
| 10 | Manually packed narrow representations | Native storage/arithmetic; error versus common fp32 reference |
| 11 | Read and write strides 1–64 | Separate read/write stride; fixed-span permutations |
| 12 | Identity/block/random/clustered gather | Duplicate rate and working-set size; correct scratch metadata |
| 13 | Atomic scatter with collision distributions | Plain unique stores; same-word versus neighboring-word atomic topology |
| 14 | Duplicate reads in different invocation schedules | Schedule distance is not ordered temporal reuse; dependent touches |
| 15 | Device-local bandwidth saturation | Warm/evicted distinction; robust plateau/uncertainty checks |
| 16 | Direct/shared stencil with reuse radii | Working set × radius × placement |
| 17 | Tile/local sizes 8/16/32/64 | Fix dispatch bounds; vary tile size independently of workgroup size |
| 18 | Temporary-state proxy and exact transform oracle | Fixed-work live-range study with register/spill evidence |
| 19 | Uniform/alternating/random branch predicates | Fixed predicate cost/probability, spatial mask coherence, compiled branch evidence |
| 20 | Flat/tiled loops and barrier intervals | Fix shared race; hold communication/ALU work constant |
| 21 | Global atomic versus shared-tree reduction | Multi-pass reduction, fan-in and elements per thread |
| 22 | Full three-dispatch inclusive scan | Refresh stale metrics; recursive hierarchy beyond one-workgroup block scan |
| 23 | Global/shared histogram with hot bins | Bin count, spacing, replication and atomic topology |
| 24 | Stable staged versus unordered atomic compaction | Fix bytes; compare equal ordering contracts |
| 25 | Original versus CPU-sorted input to identical 1D binning kernel | Include sorting cost/reuse break-even; planned broader pipelines are absent |
| 26 | Contiguous input offsets 0–64 bytes | Vector width × input/output alignment × actual subgroup width |
| 27 | Sequential/block-shuffled/random permutation | Factor within-subgroup coalescing from block order/cache residency |
| 28 | Direct host-visible versus staged device-local paths | Fix six-buffer budget; placement metadata; reuse/transfer amortization |
| 29 | Shared strides and padding, constant static footprint | Check 33 KiB requirement; broadcast, width and active-lane controls |
| 30 | Shared-tree versus subgroup-assisted block reduction | Subgroup storage limits; final atomic may dominate |
| 31 | Shared/subgroup block-local scan, items per thread | Portable order and dispatch bounds; block size/data mapping also change |
| 32 | Block-local shared-atomic/ballot compaction | Portable stability, survivor bytes, equal ordering contracts |
| 33 | Copy reference and naive/tiled/padded transpose | Rectangles, pitches, partial tiles, tile sizes, image path |
| 34 | Stable key-only 4/8-bit radix sort | Stage timing, payload stability, input distributions, scalable prefix/rank |
| 35 | Fixed useful work with declared shared footprints | Verify actual driver allocation/occupancy; shared/local-size limits |
| 36 | Compact/padded flat node checksum, sequential/hash | Same real BVH/rays, dependent traversal, nodes visited |
| 37 | Eight remapped read frames | Per-frame oracle; hold spatial locality/unique footprint constant |
| 38 | Visibility append versus append plus bin counts | Actual cluster rejection/list use and downstream savings |
| 39 | Light-centric and tile-centric counting | Remove global stats hotspot; algorithms intentionally examine different candidates |
| 40 | Static versus persistent per-lane task claims | Claim chunk size, worker count, subgroup/local queues, heavy-tail costs |
| 41 | Shared versus subgroup sum plus ballot | Portable mapping; cache-resident repeated primitive timing |
| 42 | Serial two-pass versus fused single-pass proxy | Report correctly disclaims queue overlap; implement actual queue concurrency |
| 43 | Compact SoA/padded AoS flat ray-style checksum | Equal-footprint AoS/SoA, real rays/traversal, reorder amortization |
| 44 | Staged versus fused cull/bucket/emit-style transforms | Actual compaction/indirect arguments; retain logical and modeled traffic separately |
| 45 | Deterministic coalesced/stride/hash signature on one GPU | Multi-device aggregation; equal unique source footprint; device selection/provenance |

## Measurement changes that unlock stronger experiments

**Make actual memory placement visible.** Most non-baseline kernels require `HOST_VISIBLE | HOST_COHERENT` and accept the first matching type. Those requirements do not establish host RAM or device-local memory: a selected type may have both properties. Export `memoryTypeIndex`, heap index, actual property flags, requested/allocated/live bytes, and optional memory budget. Add staged device-local controls to cache/layout experiments. Do not diagnose PCIe solely from approximately 20 GB/s observations. Vulkan distinguishes these memory properties and UMA/discrete transfer behavior: [allocation guide](https://docs.vulkan.org/guide/latest/memory_allocation.html), [memory specification](https://docs.vulkan.org/spec/latest/chapters/memory.html).

**Separate observables.** Export logical elements, useful bytes, modeled shader bytes, observed hardware bytes when available, unique footprint, and actual dispatches independently. Label every bandwidth denominator. Cached repeated reads can produce a large requested-byte GB/s without corresponding external-memory traffic. Atomic bytes are a model, not a bus measurement. Preserve stability/order semantics when comparing primitives.

**Make runs reproducible.** Add device selection, vendor/device/UUID, queue family and subgroup properties, full CLI, seed, shader/binary/source hashes, and analyzer version. Preserve per-iteration data. Use independent repeated runs and paired/randomized variant ordering, record the ordering seed, and report uncertainty across runs. Five within-run samples give a weak tail estimate and cannot establish small cross-variant differences. Keep validation/counter collections separate from uninstrumented timing collections.

**Verify proposed mechanisms.** Disassemble resource-sensitive shaders and use Vulkan-capable tooling when counters are available. NVIDIA's [Nsight Graphics GPU Trace](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-overview.html) can inspect utilization, queue synchronization and under-utilization; exact metrics depend on architecture/tool support. Use equivalent vendor tools on other devices. Cache-hit, transaction, occupancy, and spill claims should identify direct evidence versus timing-based inference. Do not assume CUDA-only profiler workflows apply to Vulkan.

## Prioritized additional experiments

These are designs to implement, not new registered experiments or measured conclusions. Prefer extending the existing ID where indicated.

### A. Dependent-load latency and memory-level parallelism — highest priority, new study

- **Question:** Does an access pattern lose performance because each dependent load is slow, or because too few loads are outstanding?
- **Vary:** Working-set size, then independent chains per invocation (1/2/4/8/16) as a separate axis. Use randomized permutation cycles and fixed seeds. Start with staged device-local memory, then repeat other placement policies.
- **Hold constant:** Total requested loads, source footprint/address distribution, participating lanes and final output size for each comparison. Shorten each chain when increasing chain count; make all chains contribute observably. Record the extra register state as an intended mechanism.
- **Measure/validate:** Dispatch ns per requested dependent load and aggregate loads/s; validate each chain's final state and rolling digest against a CPU walk. Always write the same maximum-size output record, including sentinels for inactive chain slots, so output traffic stays constant. Long dependent loops amortize launch overhead. Call this an effective dependent-load cost, not an exact isolated hardware latency. Every active chain must contribute observably; unchecked sampled reads are insufficient.

### B. Factor coalescing, spatial locality and cache residency — highest priority, extend 11/12/14/27

- **Vary:** A 2×2 access design: ordered versus permuted addresses inside aligned blocks, and sequential versus shuffled block order. Independently sweep block width, working set and reuse.
- **Hold constant:** A bijective visit to the same source elements, index-buffer bytes, arithmetic, output and total loads. Avoid a strided case with a much larger span or a hash-with-replacement case with fewer unique addresses.
- **Measure/validate:** Time and useful/requested GB/s, exact output, supported cache/transaction counters. Separate warm preparation from an eviction-buffer preparation outside the timed region. An eviction procedure is a reproducible perturbation, not a guarantee that every cache is cold. Claim cache capacity/hits only with appropriate evidence.

### C. Placement, upload/readback policy and reuse break-even — highest priority, extend 28

- **Vary:** Available actual memory types: direct host-visible, staged device-local, host-cached readback, and coherent/noncoherent variants; upload once → `K` identical dispatches → readback once for `K=1,2,4,16,64`.
- **Hold constant:** Kernel/output, useful elements, validation and total live budget. Query rather than assume distinct heaps or ReBAR/UMA behavior.
- **Measure/validate:** CPU fill, flush/invalidate, transfer, kernel, full GPU path and application wall time separately; chart the measured reuse crossover. Correctly align noncoherent ranges and fence ownership before host access. Export actual flags, heap/type, atom size and allocation bytes. Memory properties and host access tradeoffs are specified by [Khronos memory allocation](https://docs.vulkan.org/spec/latest/chapters/memory.html).

### D. Buffers versus texel buffers, sampled images and storage images — high priority, new study

- **Vary:** Resource representation for the same integer 2D dataset; row-major, column-major, neighborhood and random access. Start with a common supported 32-bit integer format and exact unfiltered fetches.
- **Hold constant:** Coordinates, fetch count, arithmetic and common output buffer. Exclude format conversion/filtering from the first comparison; add them as later explicit variables.
- **Measure/validate:** Kernel time, useful bytes and allocated footprint; exact integer CPU oracle. Report upload/layout-transition costs separately and include them in application crossover results. Query format/usage/dimension support. Images permit implementation-dependent multidimensional layouts, making this a memory path absent from the current buffer-only transpose study: [Khronos images](https://docs.vulkan.org/spec/latest/chapters/images.html).

### E. Actual queue overlap and shared-resource interference — high priority, extend 42

- **Vary:** Serial versus two-queue execution of identical independent work: memory+ALU, memory+memory, and transfer+compute. Sweep chunk sizes and read/write mix.
- **Hold constant:** Work, bytes, output, resources and synchronization correctness. Expose queue/family availability; sharing a family does not prove hardware overlap. A fused shader is a separate comparison.
- **Measure/validate:** Total makespan, each workload's slowdown and both complete outputs; use a trace to verify concurrency. Bracket a semaphore fan-out/join on a supported timing queue or another explicitly valid timing method. Do not subtract unrelated queue timestamps blindly. Handle ownership transfers where required. Include synchronization-only overhead and end-to-end wall time. See [Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html).

### F. Atomic serialization topology and shared-memory broadcast — high priority, extend 13/23/29

- **Vary:** Same word, adjacent distinct words, widely spaced words, controlled collision rate, same-address broadcast, and shared-memory padding; then atomic width/operation where supported.
- **Hold constant:** Atomic/load operation count, input distribution, arithmetic, active lanes and allocated footprint. Use an additional factorial sweep when hot-set size must change. Keep returned atomic values used/unused as an explicit axis.
- **Measure/validate:** Operations/s, exact final counters and supported conflict/transaction evidence; label byte estimates. Avoid a global statistics counter in the timed shader. Shared-bank behavior is architecture-specific; [AMD's RDNA guide](https://gpuopen.com/learn/rdna-performance-guide/) provides one vendor's bank/padding guidance, not a universal bank-count assumption.

### G. Address-translation pressure — medium priority, new study after A/B

Keep the active node count, useful bytes and dependent load count fixed while varying address spacing, virtual span and number of touched regions. Visit regions in seeded permutation order; separately sweep within-region locality. Validate the chain and export virtual span versus actual allocated/resident bytes. Query sparse granularity if sparse buffers are used; ordinary Vulkan buffers do not expose hardware page mapping. Timing knees are **translation/cache hypotheses**, not proof of a particular TLB capacity or DRAM partition mapping.

### H. Cache pollution and producer–consumer reuse — medium priority, extend 14/37

Measure a fixed victim kernel after controlled priming and an intervening aggressor of increasing footprint, plus a repeated victim-only reference. Preserve the victim's coalescing and unique footprint, use an observable oracle for every measured invocation/frame, and keep aggressor work outside victim-only timing while also reporting total application cost. This tests whether another pass destroys useful reuse. Never label a pipeline barrier as a portable cache flush. Add queue-concurrent interference only after sequential behavior is understood.

### I. Native narrow storage, scalar layout and numerical accuracy — medium priority, extend 08/09/10/26

Compare manually packed transport against feature-gated 8/16-bit storage, and fp16 storage with fp32 versus native fp16 arithmetic. Use scalar/base layouts, vector widths and alignment as separate axes. Keep logical elements, transform and launch geometry matched where possible; publish packing/decode cost. Validate implementation correctness against a format-specific oracle **and** report max/RMS error against the same fp32/high-precision reference, including accumulated iterative error. Current error residuals do not measure quantization loss. Enable only queried features; see [Khronos shader layouts](https://docs.vulkan.org/guide/latest/shader_memory_layout.html).

### J. Register spills, shared footprint and latency hiding — medium priority, redesign 18 and extend 35

Use the same algebra/output and operation count with schedules that consume values promptly or keep them live longer. Separately vary local size and declared/verified shared allocation. Record compiled resources and spills where available, then repeat a streaming kernel and a dependent-load kernel. Plot throughput against resource footprint and measured latency hiding. Source array length alone is not an occupancy measurement; ensure any intended allocation survives the driver compiler.

### K. Sparse residency, allocation policy and working-set pressure — later, new study

Compare fully backed versus sparse-backed resources holding the **same valid accessed dataset**, then explicitly vary binding churn or active resident fraction. Query and enable sparse features/queue support, synchronize rebinds, and avoid interpreting unbound reads as valid data. Measure binding CPU/GPU cost, resident bytes, request failures and access performance. Add bounded background allocations as a separate pressure axis, using available budget estimates rather than assuming heap size is free space. [Khronos sparse resources](https://docs.vulkan.org/guide/latest/sparse_resources.html) explains binding granularity/rebinding; [memory budgets](https://docs.vulkan.org/spec/latest/chapters/memory.html) are estimates that can change with system activity.

### L. Real dependent traversal and reordering amortization — later, extend 25/36/40/43

Use one BVH topology and the same ray/query set across compact/padded AoS, equal-footprint SoA, and optional quantized nodes. Preserve exact hit semantics, record nodes visited and decoded bytes, and compare identical rays reordered by seeded permutation/Morton order. Include sorting/build/reordering time and reuse count. Add chunked/subgroup task claims under the same task costs. This converts flat checksum proxies into evidence about dependent traversal and determines when preprocessing repays its memory cost.

## Suggested implementation order

1. Fix invalid accesses, synchronization, portable subgroup semantics and limit checks; add targeted tail/large-budget/resource-limit/oracle cases.
2. Correct traffic/budget/export/build-dependency defects and bind each published artifact to one run. Recollect affected data before refreshing headline claims.
3. Add placement/provenance metadata and controlled variant scheduling to the common harness.
4. Implement A, B and C first. They distinguish latency, transactions, cache reuse, and host transfer effects that current throughput rankings combine.
5. Add D, E and F for new memory paths and contention mechanisms; use G–L for focused architecture/application depth.

For each study, require one explicit hypothesis, a table of constants versus variables, a complete measured-output oracle, a legal-device support policy, documented metric formulas, and a report distinguishing observation from architectural inference.
