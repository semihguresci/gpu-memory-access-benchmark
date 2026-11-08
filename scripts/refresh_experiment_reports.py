#!/usr/bin/env python3
"""Write reports from current exports without running benchmarks or analyzers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from analysis_integrity import validate_analysis_provenance
from experiment_workflow import ExperimentWorkflowConfig, load_enabled_experiment_configs


LIMITATIONS = {
    "01": "Repeated dispatches include compute dependencies; timing includes their synchronization cost.",
    "02": "Workgroup size can change occupancy and scheduling together; timing alone does not identify either mechanism.",
    "03": "Read-only correctness observes configured probes; requested bytes are not measured external-memory traffic.",
    "04": "Repeated dispatch timing includes compute dependencies and may reuse cached data.",
    "05": "Index mappings also change invocation scheduling; repeated dispatches include compute dependencies.",
    "06": "Active field operations are matched, but physical memory transactions require compiler or profiler evidence.",
    "07": "Active field operations are matched; blocking and physical transaction effects are not independently observed.",
    "08": "These buffer layouts do not cover every feature-gated Vulkan scalar-layout mode.",
    "09": "Padding and field organization vary together, so differences do not isolate padding alone.",
    "10": "Manual packing changes transport and conversion cost. Format-specific correctness residuals and separately reported max/RMS error versus a common fp32 oracle answer different questions; native narrow arithmetic is not covered.",
    "11": "Stride changes physical span as well as access locality; GB/s is a requested-byte measure.",
    "12": "Gather patterns may differ in duplicate rate and unique working set; the row notes retain total scratch and spans.",
    "13": "Atomic distribution changes contention; modeled atomic bytes do not establish physical bus traffic.",
    "14": "Invocation scheduling is not a guarantee of ordered temporal reuse or cache residency.",
    "15": "Timestamp throughput does not distinguish cache-hot replay from external-memory bandwidth.",
    "16": "Stencil radius changes reuse and useful work; compare matched-radius direct and tiled cases.",
    "17": "Tile size and local workgroup size vary together; timing does not isolate either factor.",
    "18": "All variants execute the same 32 chains and 256 updates in batches of 4/8/16/32 live states. Source batch width does not prove register allocation, spills, or occupancy; compiler and scheduling effects remain.",
    "19": "Predicate distributions and generated branch instructions require separate controls and compiler evidence.",
    "20": "Every variant performs the same eight work phases and publishes one immutable shared seed. b0 has one mandatory publication barrier; b1/b2/b4/b8 add 8/4/2/1 barriers. This measures additional rendezvous around fixed work; immutable reads may be hoisted.",
    "21": "The global atomic and shared reduction have different communication structures; this is not a full multi-pass reduction study.",
    "22": "Throughput counts one logical input stream across three dispatches. GB/s estimates source-level storage-buffer requests, including per-invocation carry loads: 16N+1036B-1020 bytes for N inputs and B blocks. Cache, broadcast, coalescing, and shared-memory operations are not measured DRAM traffic; the hierarchy remains bounded by the block-prefix implementation.",
    "23": "Histogram skew changes contention; shared replication and bank topology are not independently controlled.",
    "24": "The staged path preserves global input order; atomic append is unordered. GB/s estimates source-level storage-buffer accesses: atomic=16V bytes, staged=4(N+4V+4B+1) bytes, for N inputs, V survivors, B blocks. Atomic read-modify-write is modeled as 8 bytes; these are not measured DRAM transfers.",
    "25": "CPU sorting is outside the measured kernel; results do not establish end-to-end sorting amortization.",
    "26": "Offset tests use the implemented access width and placement; transaction behavior needs hardware evidence.",
    "27": "Permutation changes scheduling locality; within-subgroup coalescing and cache reuse are not independently varied.",
    "28": "Six buffers remain live within the payload budget; active direct/staged paths use two/four. Allocation padding is reported separately. Actual property flags may identify the same heap for both policies. GPU time sums stage intervals and excludes CPU submission gaps; GB/s uses useful kernel payload.",
    "29": "Shared stride and padding are source access controls; hardware bank mapping and conflict counts are not measured.",
    "30": "This is a block reduction with final atomic accumulation; that atomic can dominate subgroup differences.",
    "31": "Scan scope is block-local, with portable explicit subgroup-to-logical indexing. Items per thread also change block size and data mapping; this is not a full-array scan.",
    "32": "Compaction scope is block-local. The subgroup path preserves logical order inside each block; shared atomic append is unordered. GB/s estimates survivor input/output and per-block count traffic, 4(2V+B) bytes; it excludes shared operations and is not measured DRAM traffic.",
    "33": "Transpose results cover the configured tile and shape; image paths, pitches, and arbitrary tails need separate experiments.",
    "34": "This is stable key-only radix sorting; radix width changes pass count and auxiliary traffic, with dispatch count recorded per complete sort.",
    "35": "Declared shared memory is a proxy for allocation pressure; actual allocation, occupancy, and spills need compiler/profiler evidence.",
    "36": "The BVH-style workload is a flat checksum proxy without dependent traversal or real ray queries; layout and footprint effects are combined.",
    "37": "Every measured frame contributes to the validated output. Remapping changes spatial locality and collisions as well as temporal reuse, so timing does not isolate frame coherence.",
    "38": "This is a visibility append/bin-count proxy; it does not establish downstream culling or rendering savings.",
    "39": "Strategies examine different candidate sets. GB/s is estimated source-level traffic, including strategy-specific auxiliary accesses and modeled atomic operations, rather than observed DRAM bytes; overlap totals are validated outside timing.",
    "40": "Persistent scheduling includes per-lane task-claim costs; results depend on task distribution and do not cover all queue granularities.",
    "41": "Both paths compute the same workgroup-wide sum and odd-input count, independently of subgroup mapping. This is not a device-wide reduction, and timing alone does not measure subgroup instruction latency.",
    "42": "This compares serial two-pass and fused single-pass proxies on one queue; it does not measure concurrent queues or asynchronous overlap.",
    "43": "The ray-style workload is a flat checksum proxy without dependent traversal; layout and record footprint vary together.",
    "44": "The staged/fused cull/bucket/emit-style workload is a proxy without full compaction or indirect execution. Useful payload and modeled global traffic are distinct metrics.",
    "45": "This is a deterministic access signature on one GPU; the sweep does not establish cross-device reproducibility or a universal ordering of patterns.",
    "46": "Specialized kernels interleave 1/2/4/8/16 independent pointer chains at 64 requested loads per invocation. Aggregate ns/load includes scheduling, occupancy, and cache effects; it is not isolated hardware latency.",
    "47": "All six live buffer payloads, including the fixed 64 MiB eviction buffer, fit within the CLI scratch budget; allocation padding is recorded separately. Warm cases use an unmeasured priming dispatch; evicted cases use the sweep. Both then upload reference-complement outputs before the measured dispatch, so stale results cannot pass. That common output copy and the sweep perturb cache state; neither certifies input residency. Block widths exceed the recorded device subgroup size; identity/shuffled tables have equal requested bytes. Transaction counts are not measured.",
    "48": "Both image axes are bounded by the device image limit. Results cover only R32_UINT, exact unfiltered fetches, and the implemented shapes; requested bytes are not measured physical memory traffic.",
}


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"Non-finite JSON number: {token}")


def load_export(path: Path, experiment_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    validate_export(payload, experiment_id)
    return payload


def validate_export(payload: Any, experiment_id: str) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"{experiment_id}: missing metadata object")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{experiment_id}: missing or empty rows")
    metadata = payload["metadata"]
    if "row_count" in metadata and metadata["row_count"] != len(rows):
        raise ValueError(f"{experiment_id}: metadata row_count does not match current rows")
    if metadata.get("experiment_id", experiment_id) != experiment_id:
        raise ValueError(f"{experiment_id}: metadata belongs to another experiment")

    iterations: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        prefix = f"{experiment_id}: row {index}"
        if not isinstance(row, dict) or row.get("experiment_id") != experiment_id:
            raise ValueError(f"{prefix} belongs to another experiment")
        if row.get("correctness_pass") is not True:
            raise ValueError(f"{prefix} failed correctness")
        if not isinstance(row.get("variant"), str) or not row["variant"]:
            raise ValueError(f"{prefix} has no variant")
        for field in ("problem_size", "dispatch_count", "iteration"):
            value = row.get(field)
            minimum = 0 if field == "iteration" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{prefix} has invalid {field}")
        for field in ("gpu_ms", "end_to_end_ms", "throughput", "gbps"):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{prefix} has invalid/non-finite {field}")
        key = (row["variant"], row["problem_size"], row["dispatch_count"])
        iterations[key].append(row["iteration"])

    expected = metadata.get("timed_iterations")
    for key, values in iterations.items():
        if len(values) != len(set(values)):
            raise ValueError(f"{experiment_id}: duplicate iterations for {key}")
        if expected is not None and (type(expected) is not int or expected <= 0 or sorted(values) != list(range(expected))):
            raise ValueError(f"{experiment_id}: missing iterations for {key}; expected {expected}")


def select_focus_rows(rows: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    sizes: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        sizes[(row["variant"], row["dispatch_count"])].add(row["problem_size"])
    common = set.intersection(*sizes.values())
    common_size = max(common) if common else None
    return bool(common), [
        row for row in rows
        if row["problem_size"] == (common_size if common_size is not None else max(sizes[(row["variant"], row["dispatch_count"])]))
    ]


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ").replace("`", "'")


def _link(path: Path, experiment_root: Path) -> str:
    relative = path.relative_to(experiment_root).as_posix()
    return f"[{path.name}](./{quote(relative, safe='/._-')})"


def matching_archives(payload: dict[str, Any], experiment_root: Path) -> list[Path]:
    matches = []
    for candidate in sorted((experiment_root / "runs").rglob("*.json")):
        try:
            archived = json.loads(candidate.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
        except (OSError, ValueError):
            continue
        if archived == payload:
            matches.append(candidate)
    return matches


def current_artifacts(
    experiment_root: Path, raw_path: Path, experiment_id: str, *, raw_sha256: str | None = None,
) -> tuple[list[Path], int, str]:
    """Link only explicitly bound outputs whose tools and sources still match."""
    candidates = sorted(path for path in (experiment_root / "results").rglob("*")
                        if path.is_file() and path.suffix.lower() in (".csv", ".png", ".svg"))
    tables_dir = raw_path.parent
    try:
        provenance = validate_analysis_provenance(tables_dir, experiment_id, validate_charts=True)
        if not isinstance(provenance, dict) or not provenance.get("analysis_tools"):
            raise ValueError("analysis provenance does not bind the analyzer code; regenerate analysis")
        raw_digest = raw_sha256 or hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if not any(source.get("sha256") == raw_digest for source in provenance.get("sources", [])):
            raise ValueError("analysis provenance does not include the exact current raw export; regenerate analysis")
        bound: set[Path] = set()
        results_root = (experiment_root / "results").resolve()
        for root, field, suffixes in (
            (tables_dir, "derived_tables", (".csv",)),
            (experiment_root, "derived_charts", (".png", ".svg")),
        ):
            for name in provenance.get(field, {}):
                path = (root / name).resolve()
                if not path.is_relative_to(results_root) or path.suffix.lower() not in suffixes:
                    raise ValueError(f"invalid {field} output binding: {name}")
                bound.add(path)
        current = [path for path in candidates if path.resolve() in bound]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return [], len(candidates), f"Derived table/chart links omitted: {exc}."
    omitted_count = len(candidates) - len(current)
    reason = (
        f"Omitted {omitted_count} unbound CSV/chart files. Linked derived outputs are bound by SHA-256 "
        "to verified analyzer/plot code and sources including the exact current raw export."
    )
    return current, omitted_count, reason


def render_report(config: ExperimentWorkflowConfig) -> str:
    raw_bytes = config.output_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"), parse_constant=_reject_nonfinite)
    validate_export(payload, config.id)
    rows = payload["rows"]
    metadata = payload["metadata"]
    common, focus_rows = select_focus_rows(rows)
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in focus_rows:
        grouped[(row["variant"], row["dispatch_count"], row["problem_size"])].append(row)

    lines = [
        f"# Experiment {config.id.split('_', 1)[0]} Results: {config.display_name}", "",
        "## Run status and provenance", "",
        f"- Current export: **{len(rows)}/{len(rows)} correctness checks passed**, "
        f"{len({row['variant'] for row in rows})} variants, {len({row['problem_size'] for row in rows})} problem sizes.",
        "- Report values below are recomputed directly from the linked current raw rows; no historical ranking or causal conclusion is retained.",
        "- Test status: this report generator validates exported data; build and test results are recorded only if present in the metadata below.",
        f"- Raw export SHA-256: `{hashlib.sha256(raw_bytes).hexdigest()}`.",
        "", "| Runtime metadata | Recorded value |", "|---|---|",
    ]
    for key, value in sorted(metadata.items()):
        lines.append(f"| {_cell(key)} | `{_cell(value)}` |")
    for key in sorted(payload.keys() - {"metadata", "results", "rows"}):
        lines.append(f"| {_cell(key)} | `{_cell(payload[key])}` |")

    # These are exported runtime/metric controls, not inferred from current CLI defaults.
    note_controls: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for item in str(row.get("notes", "")).split(";"):
            key, separator, value = item.partition("=")
            if separator and any(token in key for token in (
                "budget", "scratch", "revision", "seed", "gbps_basis", "gbps_mode", "traffic_model",
                "gpu_time_scope", "throughput_basis", "validation_mode", "output_contract", "payload_bytes_per_element",
                "memory_type", "heap", "property_flags", "memory_flags", "payload", "span_bytes",
                "live_", "active_", "allocated_", "physical_residency",
                "chain_", "block_width", "subgroup_size", "within_mode", "block_mode", "residency",
                "estimated_request_bytes", "grid_width", "grid_height", "eviction_buffer_bytes",
                "oracle", "output_preparation", "warm_residency_setup", "readback_scope",
                "estimated_traffic_formula", "visible_count",
            )):
                note_controls[key].add(value)
    if note_controls:
        lines.extend(["", "Exported configuration and metric definitions (all distinct values):", "",
                      "| Row-note field | Recorded values |", "|---|---|"])
        for key, values in sorted(note_controls.items()):
            lines.append(f"| {_cell(key)} | {_cell(', '.join(sorted(values)))} |")

    lines.extend(["", "## Key measurements", ""])
    if common:
        lines.append(f"All cases share the reported **problem_size={focus_rows[0]['problem_size']}**, the largest size common to every variant and dispatch-count case.")
    else:
        lines.append("**No common problem size exists. Each variant/dispatch-count case uses its own largest size; rows are not size-matched comparisons.**")
    lines.extend(["", "Dispatch counts remain separate because averaging different repetition counts would mix different GPU work.", "",
                  "| Variant | Dispatches | Problem size | Samples | Median GPU ms | Median elements/s | Median GB/s |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for (variant, dispatches, size), samples in sorted(grouped.items()):
        medians = [statistics.median(row[field] for row in samples) for field in ("gpu_ms", "throughput", "gbps")]
        lines.append(f"| {_cell(variant)} | {dispatches} | {size} | {len(samples)} | "
                     f"{medians[0]:.6f} | {medians[1]:.3f} | {medians[2]:.6f} |")

    lines.extend(["", "## Artifacts", "", f"- Current raw data: {_link(config.output_path, config.experiment_root)}."])
    archives = matching_archives(payload, config.experiment_root)
    if archives:
        lines.append("- Archives with exactly matching JSON content: " + ", ".join(_link(path, config.experiment_root) for path in archives) + ".")
    else:
        lines.append("- Matching archive: none currently exists; historical runs are not linked as evidence for this export.")
    artifacts, _, artifact_status = current_artifacts(
        config.experiment_root, config.output_path, config.id, raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    lines.extend(f"- {_link(path, config.experiment_root)}" for path in artifacts)
    lines.append(f"- {_cell(artifact_status)}")
    lines.extend(["", "## Interpretation and limitations", "",
                  "- This is an exploratory collection from one GPU/driver with a small timed sample. Timing differences alone do not establish an architectural cause, statistical significance, or a ranking on other devices.",
                  "- GB/s follows the exported experiment metric definition. Useful payload, modeled shader accesses, and physical memory traffic are different quantities; cross-experiment GB/s comparisons require matching definitions.",
                  f"- {LIMITATIONS.get(config.id.split('_', 1)[0], 'Coverage is limited to the implemented variants and selected problem sizes.')}",
                  "- Derived table/chart links require verified content provenance, including source and tool hashes; the table above uses raw rows directly.", ""])
    return "\n".join(lines)


def main() -> None:
    configs = load_enabled_experiment_configs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["all", *configs], default="all")
    parser.add_argument("--check", action="store_true", help="Validate and render reports without writing them.")
    args = parser.parse_args()
    selected = list(configs.values()) if args.experiment == "all" else [configs[args.experiment]]
    # Validate every selected export before replacing any report.
    reports = [(config, render_report(config)) for config in selected]
    for config, report in reports:
        if not args.check:
            destination = config.experiment_root / "results.md"
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
                    prefix=".results-", suffix=".tmp", delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    temporary.write(report)
                temporary_path.replace(destination)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        print(f"[ok] {'Validated' if args.check else 'Refreshed'} {config.id}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
