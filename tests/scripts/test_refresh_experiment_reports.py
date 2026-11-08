from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from analysis_integrity import write_analysis_provenance
from refresh_experiment_reports import current_artifacts, load_export, matching_archives, render_report, select_focus_rows, validate_export


EXPERIMENT_ID = "18_register_pressure_proxy"


def make_row(variant: str = "temp_4", size: int = 1024, iteration: int = 0, dispatches: int = 1) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID, "variant": variant, "problem_size": size,
        "dispatch_count": dispatches, "iteration": iteration, "correctness_pass": True,
        "gpu_ms": 1.0, "end_to_end_ms": 2.0, "throughput": 1024000.0, "gbps": 0.008192,
    }


class RefreshReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {"metadata": {"row_count": 1, "timed_iterations": 1}, "rows": [make_row()]}

    def test_rejects_another_experiments_rows(self) -> None:
        self.payload["rows"][0]["experiment_id"] = "20_barrier_synchronization_cost"
        with self.assertRaisesRegex(ValueError, "another experiment"):
            validate_export(self.payload, EXPERIMENT_ID)

    def test_rejects_failed_correctness_and_nonfinite_metrics(self) -> None:
        for field, value in (("correctness_pass", False), ("gpu_ms", float("nan")), ("gbps", float("inf"))):
            payload = copy.deepcopy(self.payload)
            payload["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_export(payload, EXPERIMENT_ID)

    def test_rejects_missing_rows_and_iterations(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty rows"):
            validate_export({"metadata": {}, "rows": []}, EXPERIMENT_ID)
        self.payload["metadata"]["timed_iterations"] = 2
        with self.assertRaisesRegex(ValueError, "missing iterations"):
            validate_export(self.payload, EXPERIMENT_ID)

    def test_selects_largest_common_size_without_mixing_dispatches(self) -> None:
        rows = [make_row("a", 1024), make_row("a", 2048), make_row("b", 1024),
                make_row("b", 4096), make_row("a", 1024, dispatches=2)]
        common, selected = select_focus_rows(rows)
        self.assertTrue(common)
        self.assertEqual({row["problem_size"] for row in selected}, {1024})
        self.assertEqual(len(selected), 3)

    def test_missing_common_size_uses_each_cases_own_largest(self) -> None:
        common, selected = select_focus_rows([make_row("a", 1024), make_row("a", 2048), make_row("b", 4096)])
        self.assertFalse(common)
        self.assertEqual({(row["variant"], row["problem_size"]) for row in selected}, {("a", 2048), ("b", 4096)})

    def test_archives_must_match_entire_current_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            exact = runs / "exact.json"
            exact.write_text(json.dumps(self.payload), encoding="utf-8")
            stale = copy.deepcopy(self.payload)
            stale["metadata"]["measurement_revision"] = "old"
            (runs / "stale.json").write_text(json.dumps(stale), encoding="utf-8")
            self.assertEqual(matching_archives(self.payload, root), [exact])

    def test_json_nonfinite_is_rejected_at_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "invalid.json"
            source.write_text(json.dumps({**self.payload, "other": float("nan")}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-finite JSON"):
                load_export(source, EXPERIMENT_ID)

    def test_report_uses_raw_rows_and_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "results" / "tables" / "benchmark_results.json"
            source.parent.mkdir(parents=True)
            self.payload["metadata"]["measurement_revision"] = "fixed-work-v2"
            self.payload["collection_provenance"] = {"source_hash": "012345"}
            self.payload["results"] = [{"median_ms": 99999}]
            source.write_text(json.dumps(self.payload), encoding="utf-8")
            config = SimpleNamespace(
                id=EXPERIMENT_ID, output_path=source, experiment_root=root, display_name="Register Pressure Proxy",
            )
            report = render_report(config)
            self.assertIn("fixed-work-v2", report)
            self.assertIn("012345", report)
            self.assertNotIn("99999", report)
            self.assertIn("| temp_4 | 1 | 1024 | 1 | 1.000000 |", report)

    def test_excludes_stale_derived_files_from_current_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results" / "tables"
            results.mkdir(parents=True)
            source = results / "benchmark_results.json"
            source.write_text(json.dumps(self.payload), encoding="utf-8")
            stale = results / "obsolete_chart.png"
            fresh = results / "current_summary.csv"
            stale.write_bytes(b"old chart")
            fresh.write_text("variant,gpu_ms\ntemp_4,1\n", encoding="utf-8")
            os.utime(stale, ns=(1_000_000_000, 1_000_000_000))
            os.utime(source, ns=(2_000_000_000, 2_000_000_000))
            os.utime(fresh, ns=(3_000_000_000, 3_000_000_000))
            analyzer = root / "analyze.py"
            analyzer.write_text("# fixture analyzer\n", encoding="utf-8")
            write_analysis_provenance(results, [source], EXPERIMENT_ID, derived_tables=[fresh], analysis_tools=[analyzer])
            self.assertEqual(current_artifacts(root, source, EXPERIMENT_ID)[:2], ([fresh], 1))
            report = render_report(SimpleNamespace(
                id=EXPERIMENT_ID, output_path=source, experiment_root=root, display_name="Register Pressure Proxy",
            ))
            self.assertIn("current_summary.csv", report)
            self.assertNotIn("obsolete_chart.png", report)
            self.assertIn("Omitted 1 unbound CSV/chart", report)


if __name__ == "__main__":
    unittest.main()
