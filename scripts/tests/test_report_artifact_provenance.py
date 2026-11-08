from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from analysis_integrity import write_analysis_provenance
from refresh_experiment_reports import render_report


class ReportArtifactProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "example"
        self.tables = self.root / "results" / "tables"
        self.charts = self.root / "results" / "charts"
        self.tables.mkdir(parents=True)
        self.charts.mkdir()
        self.raw = self.tables / "benchmark_results.json"
        self.payload = {
            "metadata": {"experiment_id": "example", "measurement_revision": "fixed", "timed_iterations": 1},
            "rows": [{
                "experiment_id": "example", "variant": "copy", "correctness_pass": True,
                "problem_size": 16, "dispatch_count": 1, "iteration": 0,
                "gpu_ms": 1.0, "end_to_end_ms": 2.0, "throughput": 1000.0, "gbps": 0.25,
            }],
        }
        self.raw.write_text(json.dumps(self.payload), encoding="utf-8")
        self.table = self.tables / "summary.csv"
        self.table.write_text("gpu_ms_median\n1.0\n", encoding="utf-8")
        self.chart = self.charts / "summary.png"
        self.chart.write_bytes(b"fixture chart")
        self.analyzer = Path(self.temporary.name) / "analyze.py"
        self.analyzer.write_text("# analyzer revision one\n", encoding="utf-8")
        self.config = SimpleNamespace(
            id="example", display_name="Example", experiment_root=self.root, output_path=self.raw,
        )

    def _bind(self, *, source=None, charts=None):
        with contextlib.redirect_stdout(io.StringIO()):
            write_analysis_provenance(
                self.tables, [source or self.raw], "example", derived_tables=[self.table],
                derived_charts=charts if charts is not None else [self.chart], analysis_tools=[self.analyzer],
            )

    def _render(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return render_report(self.config)

    def _assert_derived_omitted(self, report):
        self.assertNotIn("[summary.csv]", report)
        self.assertNotIn("[summary.png]", report)
        self.assertIn("Derived table/chart links omitted:", report)
        self.assertIn("[benchmark_results.json]", report)
        self.assertIn("1/1 correctness checks passed", report)

    def test_valid_bound_tables_and_charts_are_linked(self):
        svg = self.charts / "padding.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        self._bind(charts=[self.chart, svg])
        report = self._render()
        self.assertIn("[summary.csv](./results/tables/summary.csv)", report)
        self.assertIn("[summary.png](./results/charts/summary.png)", report)
        self.assertIn("[padding.svg](./results/charts/padding.svg)", report)
        self.assertIn("exact current raw export", report)

    def test_unbound_csv_with_fresh_mtime_is_not_linked(self):
        stale = self.tables / "stale.csv"
        stale.write_text("gpu_ms_median\n0.0001\n", encoding="utf-8")
        self._bind()
        future = self.raw.stat().st_mtime_ns + 60_000_000_000
        os.utime(stale, ns=(future, future))
        report = self._render()
        self.assertNotIn("[stale.csv]", report)
        self.assertIn("[summary.csv]", report)
        self.assertIn("Omitted 1 unbound", report)

    def test_modified_csv_with_fresh_mtime_is_not_linked(self):
        self._bind()
        self.table.write_text("gpu_ms_median\n0.0001\n", encoding="utf-8")
        future = self.raw.stat().st_mtime_ns + 60_000_000_000
        os.utime(self.table, ns=(future, future))
        self._assert_derived_omitted(self._render())

    def test_missing_manifest_omits_derived_but_keeps_raw_report(self):
        self._assert_derived_omitted(self._render())

    def test_malformed_manifest_omits_derived_but_keeps_raw_report(self):
        (self.tables / "analysis_provenance.json").write_text("[]", encoding="utf-8")
        self._assert_derived_omitted(self._render())

    def test_analyzer_hash_change_omits_derived(self):
        self._bind()
        self.analyzer.write_text("# changed metric formula\n", encoding="utf-8")
        self._assert_derived_omitted(self._render())

    def test_plotter_hash_change_omits_derived(self):
        self._bind()
        plotter = Path(self.temporary.name) / "plot.py"
        plotter.write_text("# plotter revision one\n", encoding="utf-8")
        manifest_path = self.tables / "analysis_provenance.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["plot_tools"] = {str(plotter): hashlib.sha256(plotter.read_bytes()).hexdigest()}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        plotter.write_text("# changed chart formula\n", encoding="utf-8")
        self._assert_derived_omitted(self._render())

    def test_current_source_hash_change_omits_derived_and_updates_raw_values(self):
        self._bind()
        self.payload["rows"][0]["gpu_ms"] = 7.0
        self.raw.write_text(json.dumps(self.payload), encoding="utf-8")
        report = self._render()
        self._assert_derived_omitted(report)
        self.assertIn("7.000000", report)

    def test_same_revision_archive_without_current_content_is_not_evidence(self):
        archive = self.root / "runs" / "archive.json"
        archive.parent.mkdir()
        archived = json.loads(json.dumps(self.payload))
        archived["rows"][0]["gpu_ms"] = 0.5
        archive.write_text(json.dumps(archived), encoding="utf-8")
        self._bind(source=archive)
        report = self._render()
        self._assert_derived_omitted(report)
        self.assertIn("does not include the exact current raw export", report)

    def test_chart_hash_change_omits_derived(self):
        self._bind()
        self.chart.write_bytes(b"stale chart given a fresh modification time")
        self._assert_derived_omitted(self._render())

    def test_memory_placement_and_allocation_notes_remain_visible(self):
        controls = {
            "scratch_size_bytes": "8192", "live_buffer_count": "6", "active_buffer_count": "4",
            "live_payload_bytes": "6144", "active_payload_bytes": "4096", "allocated_memory_bytes": "12288",
            "source_memory_type": "2", "source_heap": "1", "source_memory_flags": "15",
            "device_input_memory_type_index": "2", "device_input_heap_index": "1",
            "device_input_memory_property_flags": "15", "resident_buffer_payload_bytes": "6144",
            "allocated_buffer_bytes": "12288", "active_source_bytes": "1024",
        }
        self.payload["rows"][0]["notes"] = ";".join(f"{key}={value}" for key, value in controls.items())
        self.raw.write_text(json.dumps(self.payload), encoding="utf-8")
        report = self._render()
        for key, value in controls.items():
            with self.subTest(key=key):
                self.assertIn(f"| {key} | {value} |", report)


if __name__ == "__main__":
    unittest.main()
