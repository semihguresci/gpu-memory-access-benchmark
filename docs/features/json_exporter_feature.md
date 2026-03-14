# JSON Export Feature

## Source
- `src/utils/json_exporter.cpp`

## Purpose
Serializes benchmark summaries and optional per-iteration rows to a structured JSON output file.

## Schema Shape
- `schema`: name + version
- `metadata`:
  - export timestamp (UTC ISO-8601)
  - generator name
  - counts (`result_count`, `row_count`)
  - hardware/runtime details
- `results`: summarized benchmark metrics
- `rows` (optional): detailed per-iteration measurements and correctness flags

## Output Behavior
- Creates parent output directories if needed.
- Always writes pretty-printed JSON (`indent=2`).
- Provides overload for summary-only export and full export with rows + metadata.

## Robustness Notes
- Timestamp generation is platform-aware (`gmtime_s` on Windows, `gmtime_r` otherwise).
- Exporter assumes caller has already validated result correctness and run status.
