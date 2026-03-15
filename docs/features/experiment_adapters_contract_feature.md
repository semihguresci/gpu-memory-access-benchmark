# Experiment Adapter and Contract Feature

## Sources
- `src/experiments/adapters/dispatch_basics_adapter.cpp`
- `src/experiments/adapters/aos_soa_adapter.cpp`
- `include/experiments/experiment_contract.hpp`

## Purpose
Normalizes experiment-specific outputs into a single runtime contract used by the registry-driven dispatcher.

## Unified Contract
`ExperimentRunOutput` carries:
- `summary_results`
- `rows`
- `success`
- `error_message`

`ExperimentRunFn` signature:
- inputs: `VulkanContext`, `BenchmarkRunner`, `AppOptions`
- output parameter: `ExperimentRunOutput`
- return: `bool` success indicator

## Adapter Responsibilities
- Convert each experiment's native output shape into `ExperimentRunOutput`.
- Keep experiment-specific failure semantics local.
- Populate clear error messages for top-level reporting.

## Current Adapters
- Dispatch Basics adapter:
  - forwards scratch size and shader defaults
  - requires non-empty summaries
  - enforces `all_points_correct`
- AoS vs SoA adapter:
  - forwards scratch size and shader defaults
  - requires non-empty summaries
  - enforces `all_points_correct`
  - forwards row-level measurements

## Why This Matters
- `main.cpp` no longer needs experiment-specific branching.
- New experiments integrate through one adapter and one registry entry.
