# Runtime Entrypoint Feature

## Source
- `src/main.cpp`

## Purpose
Coordinates a full benchmark session: parse options, initialize Vulkan, run selected experiments, and export JSON.

## Startup Flow
1. Read enabled experiment IDs from generated registry.
2. Parse and validate CLI options with registry-backed experiment validation.
3. Initialize `VulkanContext`.
4. Require GPU timestamp support for benchmark runs.

## Experiment Dispatch Model
- Iterates `options.selected_experiment_ids`.
- Resolves each ID via `find_experiment_descriptor(...)`.
- Validates descriptor state (`enabled`, non-null run function).
- Executes through unified adapter contract:
  - input: `VulkanContext`, `BenchmarkRunner`, `AppOptions`
  - output: `ExperimentRunOutput`

## Aggregation and Export
- Merges experiment `summary_results` and detailed `rows`.
- Builds metadata (GPU name, Vulkan API version, driver, warmup/timed iterations, validation/timestamp flags).
- Exports via `JsonExporter`.

## Error Handling
- Any experiment failure aborts the run with a clear stderr message.
- Explicitly calls `context.shutdown()` on early exits.
- Catches standard and non-standard exceptions at top level.
