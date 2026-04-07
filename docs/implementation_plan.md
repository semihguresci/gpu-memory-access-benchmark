# Implementation Plan

This plan turns the current repository into a more maintainable benchmark platform and a stronger research deliverable.
It is written against the repository state on April 5, 2026.

## Objective

Deliver a platform where:

- experiment registration is manifest-driven end to end
- shared runtime code owns common mechanics
- experiments focus on benchmark-specific logic and correctness
- reports can be regenerated from raw outputs without manual repair
- the headline memory-access narrative is backed by current multi-GPU data

## Phase 1: Stabilize the Platform Contract

Status:
- partially complete

Completed foundation:
- shared manifest introduced
- CMake and Python tooling read the manifest
- representative GPU smoke tests are wired into `ctest`

Remaining work:
- audit every experiment for manifest, metadata, and scratch-budget consistency
- ensure all generated docs indexes come from manifest-driven scripts
- document adapter responsibilities and shared metadata contracts

Primary files:
- `config/experiment_manifest.json`
- `cmake/experiments_manifest.cmake`
- `scripts/experiment_manifest.py`
- `docs/architecture.md`
- `docs/features/experiment_registry_generation.md`

Exit criteria:
- no experiment registration data is duplicated in hardcoded Python or CMake lists
- every experiment row records scratch-budget semantics consistently
- docs describe the manifest as the only registration path

## Phase 2: Extract Shared Experiment Infrastructure

Status:
- next

Work items:
- add a shared experiment-run harness for warmup, timed loops, result assembly, and common notes
- introduce reusable RAII bundles for buffers, descriptor sets, and compute pipelines
- centralize row-note helpers for recurring metadata fields
- centralize correctness-result handling for pass/fail propagation

Suggested deliverables:
- `include/experiments/experiment_run_harness.hpp`
- `src/experiments/experiment_run_harness.cpp`
- `include/experiments/compute_pipeline_bundle.hpp`
- `src/experiments/compute_pipeline_bundle.cpp`
- `include/experiments/buffer_bundle.hpp`
- `src/experiments/buffer_bundle.cpp`

Exit criteria:
- new experiments can be implemented with less duplicated Vulkan setup code
- repeated adapter and experiment boilerplate is materially reduced
- shared helpers are covered by unit tests where practical

## Phase 3: Normalize Methodology Across Experiments

Status:
- next

Work items:
- review all experiments for payload accounting correctness
- standardize total scratch budget versus per-buffer budget reporting
- standardize row notes for data size, physical span, workgroup layout, and correctness status
- ensure read-only or synthetic workloads still have end-to-end correctness signals

Primary targets:
- experiments with multiple resident buffers
- experiments with padded layouts or synthetic traffic models
- experiments that currently rely on custom note formats

Exit criteria:
- metrics are comparable across variants inside a given experiment
- results exports no longer mix internal allocation limits with user-facing CLI budgets
- benchmark rows expose enough metadata for later report generation without manual interpretation

## Phase 4: Complete Reporting and Artifact Quality

Status:
- next

Work items:
- regenerate tables, charts, and `results.md` for the full experiment set from current raw data
- standardize the structure of every `results.md`
- add missing graphs to the high-impact experiments
- add profiler screenshots for experiments where timing alone is insufficient

Primary targets:
- Experiments 06, 11, 14, 15, 16, 26, and 27
- cross-links from `readme.md` and `docs/research_overview.md`

Exit criteria:
- every implemented experiment has aligned raw outputs, derived artifacts, and report text
- the top-level narrative uses current numbers, not stale placeholders
- the most important experiments have visuals that support the written findings

## Phase 5: Cross-GPU Validation

Status:
- planned

Work items:
- run the headline memory-access experiments on at least one additional GPU class
- store raw outputs per GPU with clear hardware metadata
- compare improvement ratios across architectures
- annotate results pages with architecture-specific caveats

Primary targets:
- desktop discrete GPU
- mobile-oriented GPU if available, especially Adreno

Exit criteria:
- the public-facing results distinguish stable trends from architecture-specific behavior
- the engineering insight section cites measured cross-GPU evidence

## Phase 6: Shared-Library and Research Expansion

Status:
- later

Work items:
- expand the advanced investigation track using the same platform contracts
- reuse the shared harness for more complex workloads
- introduce richer profiling workflows where timing-only analysis is not enough

Primary targets:
- advanced plans under `docs/advanced_plans/`
- capstone-style experiments that combine memory layout, parallel primitives, and rendering-adjacent workloads

Exit criteria:
- advanced experiments build on the same stable platform rather than creating parallel infrastructure

## Near-Term Backlog

### Infrastructure

- extract the shared experiment-run harness
- extract shared compute-pipeline and buffer bundles
- add tests for manifest parsing and generation scripts where practical

### Methodology

- run a final repo-wide metadata audit over all experiments
- standardize notes and reporting fields
- verify derived artifact generation for every implemented experiment

### Research Outputs

- rerun the headline experiments on current binaries
- regenerate results pages with the latest data
- add profiler screenshots and discussion for cache/coalescing/shared-memory experiments

## Working Rules

When implementing against this plan:

- start with shared contracts before adding more experiment-specific code
- keep the manifest and generated outputs aligned
- prefer extracting a reusable helper when the same Vulkan or reporting pattern appears in multiple experiments
- do not publish a performance claim unless correctness and metadata are both defensible

## Success Condition

This implementation plan succeeds when the repository feels like a coherent benchmark platform rather than a growing pile of experiments, and when the public-facing memory-optimization story is backed by current data, stable tooling, and reviewable engineering structure.
