# Development Principles and Plan

## Mission

This repository is not just a collection of isolated benchmarks.
It is a structured GPU memory-optimization research program built on Vulkan compute, with enough engineering rigor to support repeatable measurements, reviewable implementations, and defensible written conclusions.

As of April 6, 2026, the core platform includes:

- 33 implemented experiments
- manifest-driven experiment registration
- raw benchmark export plus derived artifact generation
- per-experiment result reports
- unit coverage for shared utilities
- integration smoke coverage for representative experiments

## Development Principles

### 1. Measurement Before Claims

- use GPU timestamps for kernel timing whenever supported
- separate warmup from measured iterations
- require correctness validation before reporting a performance result as valid
- treat raw JSON exports as the primary measurement record
- make caveats explicit when a conclusion depends on one GPU or one driver stack

### 2. One Source of Truth

- experiment identity and registration data belong in the shared manifest
- build, collection, artifact generation, and documentation should derive from that manifest
- avoid duplicated experiment lists or manually synchronized registries

### 3. Shared Platform, Local Experiment Logic

- shared utilities should own Vulkan setup, timing, export, and generic harness behavior
- adapters should translate shared CLI/runtime semantics into experiment config
- experiments should own correctness, resource layout, dispatch logic, and interpretation-specific notes

### 4. Explicit Resource and Synchronization Rules

- Vulkan ownership must remain explicit and leak-free
- synchronization should stay local to the operation being measured
- benchmark code should avoid accidental CPU/GPU synchronization that distorts timing
- scratch-budget semantics must be consistent across adapters, experiments, and reports

### 5. Reporting as a First-Class Deliverable

- each experiment needs raw data, derived tables, charts, and a concise `results.md`
- reports must connect measured values to interpretation, not just display charts
- visuals and profiler evidence matter for the headline experiments

## Program Structure

### Memory-Optimization Narrative

The results-critical sequence is:

1. Experiment 06: AoS vs SoA
2. Experiment 11: Coalesced vs Strided
3. Experiment 14: Read Reuse and Cache Locality
4. Experiment 15: Bandwidth Saturation Sweep
5. Experiment 16: Shared Memory Tiling
6. Experiment 26: Warp-Level Coalescing Alignment
7. Experiment 27: Cache Thrashing, Random vs Sequential
8. Experiment 28: Device-Local vs Host-Visible Heap Placement
9. Experiment 29: Shared Memory Bank Conflict Study
10. Experiment 30: Subgroup Reduction Variants
11. Experiment 31: Subgroup Scan Variants
12. Experiment 32: Subgroup Stream Compaction Variants
13. Experiment 33: 2D Locality and Transpose Study

This is the public-facing story the repository should emphasize:

- layout controls whether accesses are contiguous
- coalescing determines whether bandwidth is used efficiently
- locality decides whether caches help or get defeated
- bandwidth saturation shows where more threads stop buying throughput
- shared memory only helps when it repays its setup and synchronization cost
- alignment and thrashing experiments make the hardware behavior concrete

### Supporting Tracks

- Core track:
  - Experiments 01-25 build execution, layout, access-pattern, synchronization, and primitive-level intuition
- Extension track:
  - Experiments 26-33 strengthen the memory-access, memory-hierarchy, subgroup, and 2D-locality story
  - Canonical plans live under `docs/experiment_plans/`
- Advanced track:
  - `docs/advanced_plans/` expands into rendering-adjacent and GPU systems investigations

## Current Development Priorities

### Priority 1: Platform Consolidation

Focus:
- reduce repeated experiment boilerplate
- keep registration manifest-driven
- make budget, metadata, and reporting contracts consistent

Expected outcomes:
- easier addition of new experiments
- lower risk of drift between build, scripts, and docs
- cleaner review surface for correctness issues

### Priority 2: Shared Library Extraction

Focus:
- extract common run harness behavior
- extract reusable buffer/pipeline/descriptor helpers used repeatedly across experiments
- centralize scratch-budget metadata handling

Expected outcomes:
- less duplicated control flow
- fewer accounting inconsistencies
- clearer separation between platform code and benchmark-specific logic

### Priority 3: Reporting Quality

Focus:
- ensure every experiment has current tables, charts, and `results.md`
- strengthen the result summaries for the headline memory-access experiments
- add profiler screenshots where chart-only analysis is weak

Expected outcomes:
- stronger portfolio presentation
- easier comparison across GPU architectures
- less manual cleanup after reruns

### Priority 4: Cross-GPU Validation

Focus:
- rerun the headline experiments on more than one architecture
- compare desktop versus mobile-oriented behavior where possible
- identify which findings generalize and which are architecture-specific

Expected outcomes:
- more credible conclusions
- better engineering insight section in public-facing materials

## Milestones

### M1. Platform Integrity

- manifest-driven registration is the only supported registration path
- representative smoke tests run through `ctest`
- shared metadata semantics are stable across experiments

### M2. Shared Library Maturity

- repeated experiment scaffolding is extracted into common helpers
- new experiments can be added with minimal boilerplate
- static analysis and test coverage protect the shared layers

### M3. Results Consistency

- all existing experiments have current raw outputs, charts, and `results.md`
- reporting language is consistent across experiments
- artifact regeneration is reliable and low-friction

### M4. Headline Research Quality

- the memory-optimization narrative is fully backed by current measured data
- graphs and screenshots exist for the most important experiments
- written conclusions explain why the hardware behaves the way it does

### M5. Cross-GPU Confidence

- the critical experiments are rerun on at least two GPU classes
- reports distinguish universal findings from architecture-specific ones

## Quality Gates

Any substantial experiment or platform change should satisfy these gates:

- build passes with the repository presets
- unit tests pass
- GPU smoke tests pass when integration tests are enabled
- changed experiments keep correctness checks intact
- raw outputs and generated reports do not drift in naming or metadata
- documentation reflects the current implementation model

## Definition of Done

The memory-optimization program is in a strong state when:

- the build, scripts, and docs all derive from one shared manifest
- shared runtime utilities handle the platform concerns cleanly
- experiment implementations focus on benchmark logic rather than plumbing
- every key experiment has correctness checks, raw data, plots, and concise analysis
- the headline results can be explained with both numbers and engineering insight
- cross-GPU reruns confirm which conclusions are robust

For concrete sequencing, see [implementation_plan.md](implementation_plan.md).
