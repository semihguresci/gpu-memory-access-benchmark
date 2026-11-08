# Existing experiment integrity fixes

Date: 2026-10-23. Scope: repair the defects in the [45-experiment review](experiment_review_and_expansion_plan.md), preserving existing experiment IDs. New research studies from that review remain proposals.

## Implemented repairs

| Reviewed issue | Resolution |
|---|---|
| 01 tail writes | Logical-count push constant and bounds guard, including non-workgroup-sized tails |
| 01/02/04/05 repeated writes | Explicit compute write dependencies between repetitions, included in timing and row notes |
| 06/07 layout isolation | Matching scalar field loads/stores; common 48/80-byte useful payload, with memory placement metadata |
| 06–10 and 17/31 dispatch legality | Common problem-size caps based on the smallest legal work block |
| Descriptor/workgroup/shared limits | Storage-buffer range rejection plus manifest-driven per-experiment capability checks before execution |
| 18 register proxy | Same 32 independent chains, 256 state updates, input/output and oracle; only live-state batch scheduling varies |
| 20 shared race and changing work | Publish one immutable seed, fixed eight work phases, one mandatory publication barrier and explicitly counted additional rendezvous |
| 30/31/32 subgroup assumptions | Actual subgroup counts, sufficient scratch, dense active-lane logical indexing for ordered scan/compaction |
| 41 subgroup/local-ID assumptions | Both paths implement the same workgroup-wide sum and odd-input count, with an independent CPU oracle |
| 37 final-frame-only validation | Eight disjoint frame outputs, complement initialization, complete per-iteration validation, and a nine-span scratch budget |
| 39 observer contention | Remove per-overlap global statistics atomics; derive and validate totals from all measured tile counts on the host |
| 28 total budget | Budget all six live buffers; distinguish active/live payload, allocation padding, actual memory types and heaps |
| 24/32 traffic accounting | Survivor-sensitive source-level byte formulas, with explicit atomic RMW estimates |
| 12 scratch metadata | Preserve the original CLI total separately from per-buffer budgets and spans |
| 34 dispatch metadata | Record 12/24 physical dispatches while keeping throughput based on one logical key stream |
| Timestamp portability | Mask to valid bits, subtract modulo counter width, reject intervals whose host upper bound allows a complete wrap |
| Export failure | Detect open/write/flush/close errors and propagate a failing process exit |
| Shader include dependencies | Track recursive local includes; rebuild dependent entry points when included GLSL changes |
| 10 precision reporting | Keep format correctness residuals and add separate max/RMS error against a common fp32 reference |
| Analysis/report consistency | Reject failed/nonfinite rows, separate measurement revisions, deduplicate identical current/archive payloads, hash analysis inputs/artifacts, and regenerate reports directly from current raw rows |

Ceiling division for dispatch and packed storage counts also handles the full `uint32_t` range without overflowing an intermediate addition.

## Verification

- Release preset build succeeded with shader compilation enabled.
- CTest: **46/46 passed**, comprising 42 unit tests and four integration tests. The GPU regression matrix covers 17 representative experiments, including the repaired register/barrier studies, heap budget, compaction metrics and eight-frame oracle.
- Additional validation-layer runs covered **all 45 experiments**, **801 rows**, and complete configured output oracles. Most used `--size 4M`; radix sort used `--size 128M`. No application Vulkan VUID/synchronization errors were reported. Loader warnings about unrelated installed overlay manifests were present.
- All **100 SPIR-V modules** pass `spirv-val --target-env vulkan1.1`; their local sizes and static shared bytes match the manifest requirements exactly. Experiment 02 gates its smallest candidate and checks larger candidates individually.
- LLVM 22.1.3 clang-tidy ran for all **31 modified production translation units** with the repository configuration and changed-line filtering: no new diagnostics. The older LLVM 19 on PATH cannot parse the installed MSVC STL; the compatible Visual Studio LLVM was used instead.
- Python tests: **41 passed** across report, revision/provenance and analysis-loader regression suites. Checks include failed/missing correctness, nonfinite metrics, duplicate current/archive exports, exact-current raw inputs, explicit generated-output lists, and source/tool/table/chart tampering.
- All **45 reports** were regenerated from their current exports. Content verification covered **175 derived CSV tables, 177 PNG/SVG charts and 442 report links**; all sources and tool hashes match. Every fresh collection used the same binary and shader bundle, whose hashes still match the built files.
- Accounting checks independently reconstructed layout payloads, compaction traffic, original scratch totals, six-buffer live payloads, fixed work/barrier counts, radix dispatches/logical throughput and disjoint frame-output spans from the exported rows. The directory-as-output CLI reproduction now returns exit code 1 with an actionable export error.
- A subgroup CPU model checked 640 shuffled/sparse membership mappings, 5,760 scans, 9,600 stable compactions and 1,920 reductions, including logical tails and partial subgroups.
- Changing only the register study's common GLSL timestamp rebuilt exactly its four entry points; the following no-op shader build rebuilt zero. The source contents were not changed by this check.

## Corrected measurement collection

The fresh collection covers **all 45 experiments and 5,205 correctness-passing timed rows**. It uses measurement revision `2026-10-23-integrity-fixes`, schema `1.2.0`, five timed iterations and two warmups. Each experiment uses `32M` total scratch except radix sort (`128M`). Runs use the explicitly rebuilt Release binary on an NVIDIA GeForce RTX 2080 SUPER, driver 610.88, device API Vulkan 1.4.341.

Raw exports record the command line, binary/source/shader bundle hashes and working-tree revision. Existing April archives remain untouched and are excluded from corrected current comparisons. Derived tables/charts and per-experiment reports are regenerated from the corrected current exports; each report records its raw hash and links only a matching archived collection. Artifact manifests bind only explicitly generated CSV/PNG/SVG files to their raw sources and analyzer/plotter hashes; recent modification times alone do not establish freshness.

These are exploratory correctness-focused performance collections. Five within-run samples on one GPU do not establish statistical significance, cache-hit counts, physical DRAM bandwidth, achieved occupancy, or cross-device rankings. Flat BVH/ray-style workloads and the fused async proxy retain their documented scope. Source-level byte estimates remain distinct from observed hardware traffic.

## Reproduce

```powershell
cmake --preset windows-tests-vs
cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments gpu_memory_layout_unit_tests gpu_memory_layout_integration_tests
ctest --test-dir build-tests-vs -C Release --output-on-failure
python -m unittest discover -s tests/scripts -v
python -m unittest discover -s scripts/tests -v
python scripts/run_experiment_data_collection.py --experiment 06_aos_vs_soa --binary build-tests-vs/Release/gpu_memory_layout_experiments.exe --iterations 5 --warmup 2 --size 32M --label integrity_fixes
python scripts/generate_experiment_artifacts.py --experiment 06_aos_vs_soa
python scripts/refresh_experiment_reports.py --experiment 06_aos_vs_soa
```

Use `128M` or more when selecting radix sort. Whole-suite artifact/report generation accepts `--experiment all`; historical analysis must explicitly select the appropriate measurement revision.

## 2026-11-07 review follow-up

The follow-up repairs the new studies and their derived comparisons. Fresh runtime exports use measurement revision `2026-11-07-review-fixes`; earlier collections remain historical evidence and must not be pooled with the revised kernels.

| Reviewed issue | Resolution |
|---|---|
| 46 serialized pointer chains | Five specialized pipelines keep 1/2/4/8/16 explicit scalar addresses live and interleave their steps, retaining 64 requested loads and the original output oracle |
| 47 subgroup-local reversal | Blocks span 2/4/8/16/32 reported subgroup widths; dense active-lane membership mapping preserves bijective coverage with partial subgroups |
| 47 changing auxiliary traffic | Identity and shuffled tables both provide one index request per invocation, and both modes use the same reversal/mask arithmetic |
| 47 eviction write races | Each invocation owns one output word and reduces its disjoint sequence of eviction chunks; sweep failures propagate |
| 47 requested-byte estimate | All variants count three requested storage words per invocation (`12N` bytes); unique permutation span is separate metadata |
| 47 relative table | Match each warm variant to the sequential/ordered baseline at the same width and problem size; missing baselines fail analysis |
| 28 reuse comparison | Match host-visible baselines by problem size, dispatch count, and kernel count, retaining those counts in derived tables |
| 48 image height | Cap element count by the fixed width times `maxImageDimension2D` before allocation, alongside dispatch and integer limits |

Shader specialization inspection confirms the five chain counts expose distinct independent load addresses per loop iteration and retain exactly 64 table loads per invocation. Address and sweep coverage models check bijectivity, partial-subgroup mapping, complete eviction coverage, unique output ownership, and predicted subgroup sector sets. These checks describe source/SPIR-V behavior; they do not establish hardware transaction counts or register allocation.

LLVM 22 clang-tidy reports no diagnostics in the modified production translation units; existing included-header style diagnostics remain outside this follow-up. The installed LLVM 19 cannot parse the current MSVC standard library, so verification uses the compatible Visual Studio LLVM. Formatting and Vulkan 1.1 SPIR-V validation pass. Six new analysis regressions cover matched widths/sizes/counts, missing baselines, and legacy single-kernel rows.

The rebuilt Release binary passes all **49 CTest tests** (including the 20-experiment GPU matrix) and **54 Python tests**. Validation-layer runs cover experiments 46–48 at `4M`, plus experiment 48 at `128M`, without Vulkan VUID or synchronization errors; the loader still reports missing manifests for unrelated installed overlays.

Fresh collections on the NVIDIA GeForce RTX 2080 SUPER contain **860/860 correctness-passing timed rows**: 100 for experiment 46 (`32M`), 560 for experiment 47 (`32M`), and 200 for experiment 48 (`128M`), each with five timed iterations and two warmups. Experiment 28 retains its unchanged raw measurements and receives corrected analysis. The four affected reports and their **20 derived CSV tables and 17 charts** were regenerated and verified against current raw, tool, and artifact hashes. Earlier archives remain intact; the obsolete fixed-width timing chart is omitted from the current experiment 47 report.

## 2026-11-08 repository review fixes

| Reviewed issue | Resolution |
|---|---|
| 46/47 shader-output readback | Make compute writes visible to transfer reads before each output copy, outside compute-dispatch timing |
| 46–48 CPU readback | Make staging transfer writes visible to host reads before the existing fence wait and CPU oracle |
| 01/02 independent repetitions | Remove timestamp/row-count deduplication; retain canonical complete-payload deduplication so distinct runs exported in the same second remain separate |
| Manifest-only builds | Register the JSON manifest as a CMake configure dependency; a regression changes only enabled/workgroup/shared-memory fields and verifies automatic regeneration on build |
| 47 total scratch budget | Reserve the fixed 64 MiB eviction buffer before splitting the remaining bytes among five workload/staging buffers; export all six live payloads and allocation padding in both residency modes |
| 46 comparison chart | Label `variant time / chain_1 time` as slowdown with the matching numerator and denominator |

Experiment 47 now defaults to `128M` total scratch. Budgets of 64 MiB or less fail before allocation with an actionable size recommendation. The successful GPU regression verifies the complete six-buffer payload total against `--size`; separate cases reject `1M` and `64M` budgets.

The Release preset build, **50 CTest tests**, and **57 Python tests** pass. The three modified runtime translation units pass LLVM 22.1.3 clang-tidy with explicit C++20/MSVC and Vulkan include arguments (the Visual Studio generator does not emit a compile database); all touched C++ files pass repository formatting. Experiments 01/02 have regenerated derived artifacts and reports, with their existing raw/current/archive measurements preserved.

Fresh runtime measurements use revision `2026-11-08-review-fixes` so their corrected synchronization and experiment 47 budget are not pooled with earlier collections.

Validation-layer runs cover experiment 46 at `4M` and experiments 47/48 at `128M`, with no Vulkan VUID or synchronization errors. Fresh five-iteration, two-warmup collections contain **860/860 correctness-passing timed rows**: 100 for experiment 46 (`32M`), 560 for experiment 47 (`128M`), and 200 for experiment 48 (`128M`). Experiment 47 reports **120,992,512 bytes** of live payload within its **134,217,728-byte** CLI budget, including eviction, in every warm and evicted row.

The five affected reports (01, 02, 46, 47, 48), **43 derived tables**, and **29 charts** were regenerated and verified against their current raw, tool, and artifact hashes. Fresh raw exports have matching archives and matching binary/source provenance; historical archives remain intact. All 48 current reports pass validation, and the corrected slowdown chart was visually inspected.

## Complete repository review follow-up

The remaining review findings are repaired across the runtime and workflow:

- All 48 experiment readback paths now have explicit host-read visibility. Direct mapped-output intervals include the local compute-to-host dependency and record `gpu_time_scope`; staged copies include transfer-to-host visibility in readback and record the corresponding scope.
- Experiments 03/15 reject budgets smaller than four floats per buffer (48 total bytes), before allocation. Experiments 06–09 use a common finite-only float oracle. Scratch-size parsing rejects signs, invalid digits, and multiplication overflow while accepting the representable boundary.
- Experiment 35 complements every expected output before each dispatch. An integration regression substitutes a no-op medium variant after a correct low variant and verifies that only the no-op measurements fail. Experiments 47/48 likewise complement all outputs outside compute timing. Experiment 47 explicitly records warm priming or eviction followed by the common output upload, including its cache perturbation.
- Experiment 46 separates active payload from actual maximum-size resident buffer payload and allocation totals. Experiment 44 records visibility-dependent source-operation estimates (`32N+4V` staged, `16N` fused), with unchanged useful-payload GB/s.
- All affected collectors support external archive destinations. Single-archive analyzers select by embedded export timestamp, with filesystem time only as a fallback. GUI artifact generation requires neither an executable nor valid benchmark-only settings. Experiment 02 labels nonpower problem sizes exactly.

The Release build and **56 CTest tests** pass, including 49 unit tests, seven integration tests, and a 23-experiment GPU smoke matrix. All **64 Python tests** pass; the new workflow regressions cover every collector, every single-archive selector, both experiment 02 plotting paths, and GUI artifact-only behavior. All touched C++ files pass formatting; LLVM 22 parses the modified production translation units without compiler errors, and new diagnostics were fixed. Existing unrelated style diagnostics remain outside the changes.

Fresh measurements use revision `2026-11-08-complete-review-fixes`. Earlier collections remain historical and are not pooled with intervals or oracle setup changed by this follow-up.

Full validation-layer coverage passes **48/48 experiments and 971 measured rows**, without Vulkan VUID or synchronization errors. Probe-budget regressions additionally confirm that `12` bytes fails before unsafe operations and `48` bytes succeeds in both experiments 03/15.

Fresh collections contain **6,105/6,105 correctness-passing timed rows** on the NVIDIA GeForce RTX 2080 SUPER, with five timed iterations and two warmups. Total scratch is `32M` except experiments 34, 47, and 48 (`128M`). Every export matches its archived JSON and the common current binary, source-tree, and shader-bundle hashes. All 48 reports were regenerated and matched against the current report renderer; **192 derived tables, 191 charts, and 479 local report links** pass content/provenance checks. The experiment 02 chart's exact nonpower labels were visually checked for legibility. Historical archives remain unchanged.

## Remaining review fixes

| Reviewed issue | Resolution |
|---|---|
| 06/07 descriptor limits | Reject the experiment before resource allocation when its 8/16 storage-buffer bindings exceed either the per-stage or descriptor-set device limit |
| 01 mixed run settings | Keep measurement revision, validation mode, warmup count, and timed-iteration count separate throughout device summaries, best-dispatch selection, ratio joins, and charts |
| 22 scan traffic metric | Estimate every timed source-level storage-buffer request, including per-invocation carry loads; export the model and byte count and retain them in derived tables |
| Same-second run ordering | Export six fractional UTC digits and break legacy timestamp ties with numeric collector sequence ordering, independent of archive copy times |
| 46–48 archive-only charts | Enable plotting after archive-only analysis; real isolated generation produces four charts per experiment without a current export |
| GCC diagnostics | Explicitly initialize barrier extension pointers and remove the invalid fixed-array/null comparison |

Experiment 22 records `gbps_basis=estimated_source_level_storage_buffer_bytes` and the complete `16N+1036B-1020` byte model, where N is the logical count and B the block count. Source requests include carry reads from every invocation in each nonfirst block, including inactive tail lanes. This estimate does not measure physical DRAM transactions or compiler broadcast, cache, or coalescing effects. Logical-result throughput is unchanged.

The Release preset build and **57 CTest tests** pass: 50 unit cases and seven integration cases, including a 24-experiment GPU matrix with scan byte/time/model assertions. All **74 Python tests** pass, including configuration separation, numeric timestamp ties, fractional export ordering, traffic-model metadata, and archive-only orchestration regressions. Targeted synchronization-validation runs for experiments 06, 07, and 22 pass all 12 rows without Vulkan VUID or synchronization errors. Synthetic device-limit tests cover both storage-buffer descriptor limits and the exact 8/16 boundaries.

All touched C++ files pass repository formatting. GCC 14 accepts the six modified production translation units with `-Wall -Wextra -Wpedantic` without diagnostics. Compatible LLVM 22 clang-tidy reports no diagnostics in the changed code; pre-existing diagnostics in unrelated lines remain outside this follow-up.

Fresh measurements use revision `2026-11-08-remaining-review-fixes`, so earlier scan GB/s definitions remain historical and are not pooled with the corrected model.

Fresh collections pass **6,105/6,105 timed rows across all 48 experiments**, using five timed iterations and two warmups on the NVIDIA GeForce RTX 2080 SUPER. Scratch remains `32M` except experiments 34, 47, and 48 (`128M`). Each raw export has an exactly matching archive and the same current binary, source-tree, and shader-bundle hashes. The common binary SHA-256 is `886b3d5d10e0f6d8578f59aff625d01e9400135d11d1db44b187554f5386c120`; the source-tree SHA-256 is `799d9dd2b6f5c0b7bbea5ab30f5c8a1b587efe86fa2b431ddcf06e6c7201d6e4`.

All 48 reports were regenerated from their current exports. Their **192 derived tables, 191 charts, and 479 local links** pass content and provenance validation, and each report exactly matches the current renderer. The updated experiment 01 comparison labels and experiment 22 estimated-request chart were visually checked for legibility. Historical archives remain intact.
