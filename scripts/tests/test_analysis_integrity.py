from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from analysis_integrity import (
    LEGACY_REVISION,
    measurement_revision,
    latest_compatible_run,
    preferred_run_id,
    select_compatible_runs,
    validated_rows,
    validate_analysis_provenance,
    validate_derived_table,
    write_analysis_provenance,
    write_plot_provenance,
)


class AnalysisIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _write_run(self, name, revision=None, exported_at="2026-10-23T12:00:00Z"):
        metadata = {"exported_at_utc": exported_at}
        if revision is not None:
            metadata["measurement_revision"] = revision
        path = self.root / name
        path.write_text(json.dumps({"metadata": metadata, "rows": [self._row()]}), encoding="utf-8")
        return path

    @staticmethod
    def _row():
        return {
            "experiment_id": "example",
            "correctness_pass": True,
            "gpu_ms": 1.0,
            "end_to_end_ms": 2.0,
            "throughput": 1000.0,
            "gbps": 0.25,
        }

    def test_current_revision_excludes_archives_even_when_newer(self):
        current = self._write_run("current.json", "fixed", "2026-10-23T12:00:00Z")
        compatible = self._write_run("fixed.json", "fixed", "2026-10-23T13:00:00Z")
        legacy = self._write_run("legacy.json", exported_at="2026-10-24T12:00:00Z")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            selected = select_compatible_runs([legacy, compatible, current], current_path=current)
        self.assertEqual(selected, [current, compatible])
        self.assertIn("excluded 1 incompatible", output.getvalue())

    def test_explicit_legacy_and_legacy_default_remain_supported(self):
        current = self._write_run("fixed.json", "fixed")
        legacy = self._write_run("legacy.json")
        self.assertEqual(select_compatible_runs([current, legacy], revision=LEGACY_REVISION), [legacy])
        self.assertEqual(select_compatible_runs([legacy]), [legacy])
        self.assertEqual(measurement_revision({}), LEGACY_REVISION)

    def test_newest_archive_selects_one_revision_without_current(self):
        legacy = self._write_run("legacy.json", exported_at="2026-04-23T12:00:00Z")
        fixed = self._write_run("fixed.json", "fixed")
        self.assertEqual(select_compatible_runs([legacy, fixed]), [fixed])

    def test_same_second_legacy_archives_use_numeric_run_sequence(self):
        earlier = self._write_run("20261023_120000Z_run9.json", "fixed")
        later = self._write_run("20261023_120000Z_run10.json", "fixed")
        payload = json.loads(later.read_text())
        payload["rows"][0]["gpu_ms"] = 2.0
        later.write_text(json.dumps(payload))
        # Copy/touch times must not override the archive sequence.
        os.utime(earlier, (2_000_000_000, 2_000_000_000))
        os.utime(later, (1_000_000_000, 1_000_000_000))
        for paths in ([later, earlier], [earlier, later]):
            with self.subTest(paths=paths):
                self.assertEqual(latest_compatible_run(paths), later)

    def test_fractional_export_time_precedes_archive_name_and_copy_time(self):
        earlier = self._write_run("run10.json", "fixed", "2026-10-23T12:00:00.123450Z")
        later = self._write_run("run9.json", "fixed", "2026-10-23T12:00:00.123451Z")
        os.utime(earlier, (2_000_000_000, 2_000_000_000))
        self.assertEqual(latest_compatible_run([earlier, later]), later)

    def test_same_second_sequence_also_selects_measurement_revision(self):
        earlier = self._write_run("run9.json", "old")
        later = self._write_run("run10.json", "new")
        self.assertEqual(select_compatible_runs([later, earlier]), [later])

    def test_identical_archive_is_deduplicated_in_favor_of_current(self):
        current = self._write_run("current.json", "fixed")
        archive = self.root / "archive.json"
        payload = json.loads(current.read_text())
        # Formatting and object-key order do not make another physical run.
        archive.write_text(json.dumps(dict(reversed(list(payload.items()))), indent=4))
        with contextlib.redirect_stdout(io.StringIO()) as message:
            selected = select_compatible_runs([archive, current], current_path=current)
        self.assertEqual(selected, [current])
        self.assertIn("deduplicated 1 identical payloads", message.getvalue())

    def test_distinct_run_metadata_or_rows_are_not_deduplicated(self):
        current = self._write_run("current.json", "fixed")
        later = self._write_run("later.json", "fixed", "2026-10-23T12:00:01Z")
        changed_row = self.root / "changed_row.json"
        payload = json.loads(current.read_text())
        payload["rows"][0]["gpu_ms"] = 1.25
        changed_row.write_text(json.dumps(payload))
        self.assertEqual(
            select_compatible_runs([later, changed_row, current], current_path=current),
            [current, later, changed_row],
        )

    def test_missing_explicit_revision_fails(self):
        with self.assertRaisesRegex(ValueError, "No runs match"):
            select_compatible_runs([self._write_run("legacy.json")], revision="absent")

    def test_failed_and_missing_correctness_never_enter_rankings(self):
        for flag in (False, "false", 0, None, "unknown"):
            with self.subTest(flag=flag):
                row = self._row() | {"correctness_pass": flag}
                with self.assertRaisesRegex(ValueError, "correctness"):
                    validated_rows({"rows": [row]}, "example", self.root / "run.json")

    def test_nonfinite_metrics_reject_run(self):
        for metric in ("gpu_ms", "end_to_end_ms", "throughput", "gbps"):
            for value in (math.nan, math.inf, None, "NaN", False):
                with self.subTest(metric=metric, value=value):
                    with self.assertRaisesRegex(ValueError, metric):
                        validated_rows({"rows": [self._row() | {metric: value}]}, "example", self.root / "run.json")

    def test_other_experiment_rows_do_not_reject_selected_experiment(self):
        rows = [self._row(), self._row() | {"experiment_id": "other", "correctness_pass": False}]
        self.assertEqual(validated_rows({"rows": rows}, "example", self.root), [self._row()])

    def test_provenance_records_exact_revision_and_hash(self):
        source = self._write_run("fixed.json", "fixed")
        output = self.root / "results" / "tables"
        write_analysis_provenance(output, [source], "example")
        provenance = json.loads((output / "analysis_provenance.json").read_text())
        self.assertEqual(provenance["measurement_revision"], "fixed")
        self.assertEqual(len(provenance["sources"][0]["sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            write_analysis_provenance(output, [source, self._write_run("legacy.json")], "example")

    def test_changed_source_and_derived_table_are_rejected_before_plotting(self):
        source = self._write_run("fixed.json", "fixed")
        output = self.root / "results" / "tables"
        output.mkdir(parents=True)
        table = output / "summary.csv"
        table.write_text("gpu_ms_median\n1.0\n")
        write_analysis_provenance(output, [source], "example", derived_tables=[table])
        validate_analysis_provenance(output, "example")
        table.write_text("gpu_ms_median\n0.5\n")
        with self.assertRaisesRegex(ValueError, "derived table changed"):
            validate_analysis_provenance(output, "example")
        write_analysis_provenance(output, [source], "example", derived_tables=[table])
        source.write_text(source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            validate_analysis_provenance(output, "example")

    def test_omitted_output_list_never_blesses_obsolete_csv(self):
        source = self._write_run("fixed.json", "fixed")
        output = self.root / "results" / "tables"
        output.mkdir(parents=True)
        stale = output / "obsolete_summary.csv"
        stale.write_text("gpu_ms_median\n0.001\n")
        write_analysis_provenance(output, [source], "example")
        provenance = json.loads((output / "analysis_provenance.json").read_text())
        self.assertEqual(provenance["derived_tables"], {})
        with self.assertRaisesRegex(ValueError, "not generated"):
            validate_derived_table(stale)

    def test_new_plot_pass_does_not_rebind_obsolete_charts(self):
        source = self._write_run("fixed.json", "fixed")
        output = self.root / "results" / "tables"
        charts = self.root / "results" / "charts"
        charts.mkdir(parents=True)
        analysis_chart = charts / "analysis.svg"
        obsolete = charts / "obsolete.png"
        current = charts / "current.png"
        for path in (analysis_chart, obsolete, current):
            path.write_text("fixture")
        write_analysis_provenance(output, [source], "example", derived_charts=[analysis_chart])
        write_plot_provenance(output, "example", [obsolete])
        write_plot_provenance(output, "example", [current])
        provenance = validate_analysis_provenance(output, "example", validate_charts=True)
        self.assertEqual(set(provenance["derived_charts"]), {"results/charts/analysis.svg", "results/charts/current.png"})

    def test_archive_only_plot_is_explicitly_historical(self):
        source = self._write_run("legacy.json")
        output = self.root / "results" / "tables"
        output.mkdir(parents=True)
        current = output / "benchmark_results.json"
        current.write_text(json.dumps({"metadata": {"measurement_revision": "fixed"}}))
        write_analysis_provenance(output, [source], "example")
        with contextlib.redirect_stdout(io.StringIO()) as message:
            validate_analysis_provenance(output, "example")
        self.assertIn("historical archive revision=legacy-unversioned", message.getvalue())

    def test_preferred_run_id_uses_current_before_archive_timestamp(self):
        import pandas as pd

        index = pd.DataFrame(
            [{"run_id": "current.json", "exported_at_dt": 1}, {"run_id": "newer.json", "exported_at_dt": 2}]
        )
        self.assertEqual(preferred_run_id(index, self.root / "current.json", self.root), "current.json")
        self.assertEqual(preferred_run_id(index, None, self.root), "newer.json")

    def test_preferred_run_id_breaks_timestamp_ties_by_numeric_sequence(self):
        import pandas as pd

        for timestamp in (pd.Timestamp("2026-10-23T12:00:00Z"), pd.NaT):
            index = pd.DataFrame([
                {"run_id": "run10.json", "exported_at_dt": timestamp},
                {"run_id": "run9.json", "exported_at_dt": timestamp},
            ])
            self.assertEqual(preferred_run_id(index, None, self.root), "run10.json")

    def test_scalar_error_versus_fp32_is_separate_and_missing_stays_unknown(self):
        import pandas as pd

        script = SCRIPTS_DIR.parent / "experiments/10_scalar_type_width_sweep/scripts/analyze_scalar_type_width_sweep.py"
        spec = importlib.util.spec_from_file_location("scalar_type_analysis", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        path = self.root / "scalar.json"
        row = self._row() | {
            "experiment_id": "10_scalar_type_width_sweep",
            "variant": "u8",
            "problem_size": 16,
            "dispatch_count": 1,
            "iteration": 0,
            "notes": "max_abs_error=0;mean_abs_error=0;max_abs_error_vs_fp32=0.004;rms_error_vs_fp32=0.002",
        }
        module.INPUT_JSON = path
        path.write_text(json.dumps({"rows": [row]}))
        frame, _ = module._load_rows()
        summary = module._build_summary(frame)
        self.assertEqual(summary.iloc[0]["max_abs_error_median"], 0.0)
        self.assertEqual(summary.iloc[0]["max_abs_error_vs_fp32_max"], 0.004)
        row["notes"] = "max_abs_error=0;mean_abs_error=0"
        path.write_text(json.dumps({"rows": [row]}))
        frame, _ = module._load_rows()
        self.assertTrue(pd.isna(frame.iloc[0]["max_abs_error_vs_fp32"]))
        self.assertFalse(module._build_summary(frame).iloc[0]["error_vs_fp32_available"])


if __name__ == "__main__":
    unittest.main()
