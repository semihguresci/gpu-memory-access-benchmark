# AGENTS.md

## Purpose
Guidelines for contributors and coding agents working on this repository.
Target: production-quality Vulkan compute code in modern C++.

## Core Standards
- Use C++20, keep code warning-clean on MSVC (`/W4`) and GCC/Clang (`-Wall -Wextra -Wpedantic`).
- Prefer deterministic, explicit code over "clever" abstractions.
- Keep APIs small and focused; favor composition over inheritance.
- Avoid hidden global state.

## Formatting and Static Analysis Tooling
- The repository-level `.clang-format` and `.clang-tidy` files are the source of truth for style and static analysis configuration.
- Apply formatting to all touched C++ files in `include/` and `src/` before finalizing changes.
- Run `clang-tidy` for modified translation units whenever compile commands are available.
- Treat new `clang-tidy` diagnostics in changed code as actionable; fix them or document a concrete reason when suppression is required.
- Keep tooling changes scoped and intentional: do not reformat unrelated files in the same change.

## Build and Preset Workflow
- Prefer CMake presets over ad-hoc build directories.
- For Visual Studio test/development runs, use:
  - `cmake --preset windows-tests-vs`
  - `cmake --build --preset tests-vs-release --target gpu_memory_layout_experiments`
- `windows-tests-vs` must keep shader auto-compilation enabled (`BUILD_SHADERS=ON`).
- Shader sources are organized under `shaders/<experiment_id>/` (for example `shaders/03_memory_copy_baseline/`).
- Shader compilation is recursive under `shaders/`; keep one canonical source file per shader and avoid duplicate shader basenames across experiments (compiled outputs are emitted as `<name>.spv` in `build*/shaders/`).
- Before collecting benchmark data, ensure the selected binary has been rebuilt after experiment registration changes.
- If `scripts/run_experiment_data_collection.py` resolves a stale binary, provide `--binary` explicitly.

## Experiment Results Workflow
- For each experiment, keep `experiments/<id>/results.md` as a concise, data-backed run report.
- Include:
  - run/test status
  - hardware and run configuration metadata
  - key measured values from generated CSV/JSON outputs
  - short interpretation with explicit limitations
- Link to generated charts/tables under `experiments/<id>/results/`.
- When charts/tables are regenerated, update `results.md` so referenced artifacts match current files and remove stale entries.


## File and Type Structure
- Use one primary public type per header/source pair.
- Keep declarations in `include/`, definitions in `src/`.
- Core Vulkan utility types (for example `GpuTimestampTimer`) must always have dedicated `.hpp` and `.cpp` files.
- Keep public headers lightweight; include only what is required.
- Use clear naming:
  - `PascalCase` for types.
  - `snake_case` for variables and functions.
  - `kPascalCase` for `constexpr` constants.

## Vulkan Engineering Rules
- Check every Vulkan call that returns `VkResult`.
- Destroy Vulkan objects in reverse creation order.
- Reset handles to `VK_NULL_HANDLE` after destruction.
- Keep ownership explicit: the creator is responsible for teardown unless documented otherwise.
- Never assume timestamp support; validate queue family capabilities before use.
- Record command buffers with clear phases:
  - resource/state setup
  - dispatch/draw
  - synchronization and readback
- Keep synchronization explicit and local to the operation being measured or executed.

## Error Handling and Logging
- Return `bool`/status for recoverable failures, and log actionable messages to `std::cerr`.
- Do not swallow errors from Vulkan/IO operations.
- Keep failure paths simple and leak-free.
- Keep progress/process logs on `std::cout` gated behind the runtime flag `--verbose-progress`.
- Default runs should stay quiet (no per-iteration progress spam) unless `--verbose-progress` is explicitly enabled.
- New experiment configs should carry `verbose_progress` and adapters must forward `AppOptions::verbose_progress`.
- `scripts/run_experiment_data_collection.py` supports `--verbose-progress`; keep script and binary flag behavior aligned.

## Performance and Safety
- Minimize allocations in per-dispatch paths.
- Avoid unnecessary CPU/GPU synchronization in benchmark code.
- Use timestamp queries or fences intentionally; document what is being measured.
- Validate assumptions with Vulkan validation layers during development.

## Review Checklist
- Builds successfully with current `CMakeLists.txt`.
- No resource leaks (buffers, memory, descriptor sets/layouts, pipelines, query pools, fences, command pools).
- New code follows existing style and naming.
- Public interfaces are documented by clear names and straightforward signatures.
