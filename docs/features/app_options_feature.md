# CLI and Experiment Selection Feature

## Source
- `src/utils/app_options.cpp`

## Purpose
Parses runtime options and validates experiment selection against the generated registry.

## Supported Options
- `--validation`
- `--experiment`
- `--iterations`
- `--warmup`
- `--size`
- `--output`

## Selection Semantics
- `--experiment all`: selects all enabled experiments provided by registry.
- `--experiment id1,id2,...`: comma-separated list with whitespace trimming.
- Duplicate IDs are deduplicated while preserving first-seen order.

## Validation Rules
- Fails if registry has no enabled experiments.
- Fails for empty selection token.
- Fails for unknown experiment ID.
- Fails for invalid size format.
- Normalizes output path to `.json` suffix.

## Size Parsing
- Accepts bytes or suffix forms: `K`, `M`, `G`.
- Examples: `4194304`, `4M`, `1G`.

## Failure Behavior
- Prints actionable stderr messages.
- Exits with status code `2` for argument errors.
