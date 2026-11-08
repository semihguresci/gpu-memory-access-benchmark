"""Shared correctness and measurement-version checks for benchmark analysis."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

LEGACY_REVISION = "legacy-unversioned"


def measurement_revision(metadata: dict) -> str:
    return str(metadata.get("measurement_revision") or LEGACY_REVISION)


def validated_rows(payload: dict, experiment_id: str, source: Path) -> list[dict]:
    """Reject an entire input run before ranking any failed or invalid measurement."""
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: benchmark payload must be an object.")
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"{source}: rows must be a list.")
    selected = [row for row in rows if isinstance(row, dict) and row.get("experiment_id") == experiment_id]
    for index, row in enumerate(selected):
        correctness = row.get("correctness_pass")
        passed = correctness is True or (
            isinstance(correctness, (str, int)) and str(correctness).strip().lower() in {"1", "true", "yes"}
        )
        if not passed:
            raise ValueError(f"{source}: row {index} failed or lacks correctness; performance analysis refused.")
        for metric in ("gpu_ms", "end_to_end_ms", "throughput", "gbps"):
            value = row.get(metric)
            try:
                finite = not isinstance(value, bool) and math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                raise ValueError(f"{source}: row {index} has invalid {metric}; performance analysis refused.")
    return selected


def _run_timestamp(path: Path, payload: dict) -> float:
    timestamp = str(payload.get("metadata", {}).get("exported_at_utc", ""))
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
    except (ValueError, OverflowError, OSError):
        return path.stat().st_mtime


def _natural_run_key(run_id: str) -> tuple:
    """Break legacy timestamp ties by collector sequence, independent of copy times."""
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", run_id))


def _run_order_key(path: Path, payload: dict) -> tuple:
    return _run_timestamp(path, payload), _natural_run_key(path.as_posix())


def select_compatible_runs(
    paths: Iterable[Path], *, current_path: Path | None = None, revision: str | None = None
) -> list[Path]:
    """Select one revision, preferring current metadata; explicit legacy analysis remains supported."""
    candidates: list[tuple[Path, dict]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] Skipping unreadable JSON: {path} ({exc})")
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list) or not payload["rows"]:
            print(f"[warn] Skipping JSON without benchmark rows: {path}")
            continue
        candidates.append((path, payload))
    if not candidates:
        return []
    current = next(
        ((path, payload) for path, payload in candidates if current_path and path.resolve() == current_path.resolve()),
        None,
    )
    if revision is None:
        reference = current or max(candidates, key=lambda candidate: _run_order_key(*candidate))
        revision = measurement_revision(reference[1].get("metadata", {}))
    selected = [path for path, payload in candidates if measurement_revision(payload.get("metadata", {})) == revision]
    # Current wins content de-duplication and supplies the current-run summary even if archives are newer.
    selected.sort(key=lambda path: 0 if current_path and path.resolve() == current_path.resolve() else 1)
    excluded = len(candidates) - len(selected)
    payloads = {path.resolve(): payload for path, payload in candidates}
    seen_payloads: set[str] = set()
    unique: list[Path] = []
    for path in selected:
        canonical = json.dumps(payloads[path.resolve()], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if digest in seen_payloads:
            print(f"[info] Skipping duplicate run payload: {path}")
            continue
        seen_payloads.add(digest)
        unique.append(path)
    duplicate_count = len(selected) - len(unique)
    selected = unique
    print(f"[info] measurement_revision={revision}; selected {len(selected)} runs; excluded {excluded} incompatible runs; "
          f"deduplicated {duplicate_count} identical payloads.")
    if not selected:
        raise ValueError(f"No runs match measurement_revision={revision!r}.")
    return selected


def preferred_run_id(run_index, current_path: Path | None, experiment_root: Path) -> str:
    if current_path:
        try:
            current_id = current_path.resolve().relative_to(experiment_root.resolve()).as_posix()
        except ValueError:
            current_id = current_path.name
        if current_id in set(run_index["run_id"]):
            return current_id
    newest_timestamp = run_index["exported_at_dt"].max()
    newest = run_index[run_index["exported_at_dt"] == newest_timestamp]
    run_ids = newest["run_id"] if not newest.empty else run_index["run_id"]
    return str(max(run_ids, key=_natural_run_key))


def latest_compatible_run(paths: Iterable[Path]) -> Path | None:
    """Choose the newest compatible export independently of archive copy times."""
    selected = select_compatible_runs(paths)
    if not selected:
        return None
    return max(
        selected,
        key=lambda path: _run_order_key(path, json.loads(path.read_text(encoding="utf-8"))),
    )


def format_problem_size(value: int) -> str:
    """Use power notation only for exact powers; retain arbitrary element counts."""
    count = int(value)
    return f"2^{count.bit_length() - 1}" if count > 0 and count & (count - 1) == 0 else f"{count:,}"


def validate_summary_table(frame, source: Path) -> None:
    if "correctness_pass_rate" in frame:
        if any(not math.isfinite(float(value)) or float(value) != 1.0 for value in frame["correctness_pass_rate"]):
            raise ValueError(f"{source}: summary includes incorrect measurements; plotting refused.")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_records(paths: Iterable[Path], root: Path) -> dict[str, str]:
    records = {}
    for path in paths:
        if path is None:
            continue
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"{path}: declared analysis output/tool does not exist.")
        try:
            name = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            name = str(path)
        records[name] = _file_hash(path)
    return records


def _validate_tool_records(records: dict[str, str], manifest: Path) -> None:
    if not records:
        raise ValueError(f"{manifest}: missing tool provenance; rerun analysis.")
    repository_root = Path(__file__).resolve().parent.parent
    for name, digest in records.items():
        path = Path(name)
        if not path.is_absolute():
            path = repository_root / path
        if not path.is_file() or _file_hash(path) != digest:
            raise ValueError(f"{path}: analysis/plot tool changed; regenerate derived artifacts.")


def validate_derived_table(source: Path) -> None:
    """Require the exact requested CSV to be an explicitly generated, unchanged output."""
    manifest = source.parent / "analysis_provenance.json"
    if not manifest.is_file():
        raise ValueError(f"{manifest}: missing provenance; rerun analysis before reading derived tables.")
    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    digest = provenance.get("derived_tables", {}).get(source.name)
    if digest is None:
        raise ValueError(f"{source}: table was not generated by the recorded analysis; rerun analysis.")
    if not source.is_file() or _file_hash(source) != digest:
        raise ValueError(f"{source}: derived table changed after analysis; rerun analysis before plotting.")


def validate_analysis_provenance(
    tables_dir: Path, experiment_id: str, *, validate_charts: bool = False, required_source: Path | None = None
) -> dict:
    """Prevent derived-data plots from silently displaying stale or mixed source revisions."""
    manifest = tables_dir / "analysis_provenance.json"
    if not manifest.exists():
        raise ValueError(f"{manifest}: missing provenance; rerun the experiment analyzer before plotting.")
    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    if provenance.get("provenance_version") != 2:
        raise ValueError(f"{manifest}: obsolete output provenance; rerun analysis.")
    if provenance.get("experiment_id") != experiment_id:
        raise ValueError(f"{manifest}: experiment mismatch; rerun analysis.")
    _validate_tool_records(provenance.get("analysis_tools", {}), manifest)
    revision = provenance.get("measurement_revision")
    sources = provenance.get("sources", [])
    if not sources:
        raise ValueError(f"{manifest}: missing source files; rerun analysis.")
    experiment_root = tables_dir.parent.parent
    for source in sources:
        path = Path(source["path"])
        if not path.is_absolute():
            path = experiment_root / path
        if not path.is_file() or _file_hash(path) != source.get("sha256"):
            raise ValueError(f"{path}: source changed after analysis; rerun analysis before plotting.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if measurement_revision(payload.get("metadata", {})) != revision:
            raise ValueError(f"{path}: measurement revision mismatch; rerun analysis.")
        if not validated_rows(payload, experiment_id, path):
            raise ValueError(f"{path}: no rows for {experiment_id}; rerun analysis.")
    if required_source is not None and _file_hash(required_source) not in {source["sha256"] for source in sources}:
        raise ValueError(f"{required_source}: current raw export is not the analyzed source; rerun analysis.")
    for name in provenance.get("derived_tables", {}):
        validate_derived_table(tables_dir / name)
    if validate_charts and provenance.get("derived_charts"):
        _validate_tool_records(provenance.get("plot_tools", {}), manifest)
        for name, digest in provenance["derived_charts"].items():
            path = experiment_root / name
            if not path.is_file() or _file_hash(path) != digest:
                raise ValueError(f"{path}: derived chart changed after generation; rerun artifact generation.")
    current = tables_dir / "benchmark_results.json"
    if current.exists():
        current_revision = measurement_revision(json.loads(current.read_text(encoding="utf-8")).get("metadata", {}))
        if current_revision != revision:
            if provenance.get("source_selection") != "archive_only":
                raise ValueError(f"{manifest}: current revision differs; rerun analysis.")
            print(f"[info] Plotting historical archive revision={revision}; current revision={current_revision}.")
    print(f"[info] Verified derived artifacts for measurement_revision={revision}.")
    return provenance


def write_analysis_provenance(
    tables_dir: Path, paths: Iterable[Path], experiment_id: str, *,
    derived_tables: Iterable[Path] = (), derived_charts: Iterable[Path] = (), analysis_tools: Iterable[Path] = ()
) -> None:
    paths = list(paths)
    sources = []
    revisions = set()
    for path in paths:
        data = path.read_bytes()
        payload = json.loads(data)
        revision = measurement_revision(payload.get("metadata", {}))
        revisions.add(revision)
        if not validated_rows(payload, experiment_id, path):
            raise ValueError(f"{path}: no rows for {experiment_id}.")
        try:
            source_path = path.resolve().relative_to(tables_dir.parent.parent.resolve()).as_posix()
        except ValueError:
            source_path = str(path.resolve())
        sources.append({"path": source_path, "sha256": hashlib.sha256(data).hexdigest()})
    if len(revisions) != 1:
        raise ValueError("Analysis provenance requires exactly one measurement revision.")
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_records = _file_records(derived_tables, tables_dir)
    if any(Path(name).name != name or name.startswith("benchmark_results") for name in table_records):
        raise ValueError("Declared derived tables must be non-raw files directly inside the tables directory.")
    chart_records = _file_records(derived_charts, tables_dir.parent.parent)
    tool_records = _file_records([Path(__file__), *analysis_tools], Path(__file__).resolve().parent.parent)
    provenance = {
        "provenance_version": 2,
        "experiment_id": experiment_id,
        "measurement_revision": next(iter(revisions)),
        "correctness_policy": "reject_failed_missing_or_nonfinite_measurements",
        "sources": sources,
        "source_selection": "includes_current"
        if any(path.resolve() == (tables_dir / "benchmark_results.json").resolve() for path in paths)
        else "archive_only",
        "derived_tables": table_records,
        "derived_charts": chart_records,
        "analysis_charts": chart_records,
        "analysis_tools": tool_records,
        "plot_tools": tool_records if chart_records else {},
    }
    (tables_dir / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"[info] analysis measurement_revision={provenance['measurement_revision']}; {len(sources)} source files.")


def write_plot_provenance(
    tables_dir: Path, experiment_id: str, charts: Iterable[Path], *, plot_tools: Iterable[Path] = ()
) -> None:
    provenance = validate_analysis_provenance(tables_dir, experiment_id)
    # A new plot pass replaces its prior output list; optional/renamed old charts stay unbound.
    analysis_charts = provenance.get("analysis_charts", {})
    provenance["derived_charts"] = {**analysis_charts, **_file_records(charts, tables_dir.parent.parent)}
    provenance["plot_tools"] = {
        **(provenance["analysis_tools"] if analysis_charts else {}),
        **_file_records([Path(__file__), *plot_tools], Path(__file__).resolve().parent.parent),
    }
    (tables_dir / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
