"""End-to-end guards for the raw-JSON and derived-CSV analysis entry points."""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOTS = sorted(path for path in (ROOT / "experiments").iterdir()
                          if path.is_dir() and 16 <= int(path.name[:2]) <= 45)


def load_module(path: Path):
    name = "guard_test_" + path.parent.parent.name + "_" + path.stem
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class FoundationAnalysisLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzers = [
            load_module(next((ROOT / "experiments" / experiment / "scripts").glob("analyze_*.py")))
            for experiment in ("01_dispatch_basics", "02_local_size_sweep")
        ]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @staticmethod
    def payload(experiment_id, gpu_ms):
        return {
            "metadata": {
                "measurement_revision": "fixed",
                "exported_at_utc": "2026-11-08T12:00:00Z",
                "gpu_name": "test_gpu",
                "driver_version": "1",
                "vulkan_api_version": "1.2",
                "timed_iterations": 1,
            },
            "rows": [{
                "experiment_id": experiment_id,
                "variant": "contiguous_write_ls64" if experiment_id.startswith("02_") else "contiguous_write",
                "problem_size": 16384,
                "dispatch_count": 1,
                "iteration": 0,
                "correctness_pass": True,
                "gpu_ms": gpu_ms,
                "end_to_end_ms": gpu_ms + 1.0,
                "throughput": 16384000.0 / gpu_ms,
                "gbps": 0.065536 / gpu_ms,
            }],
        }

    def test_discovery_and_loading_preserve_distinct_runs_exported_in_same_second(self):
        for analyzer in self.analyzers:
            with self.subTest(experiment=analyzer.EXPERIMENT_ID):
                runs_dir = self.root / analyzer.EXPERIMENT_ID
                runs_dir.mkdir()
                current = runs_dir / "current.json"
                repetition = runs_dir / "repetition.json"
                current.write_text(json.dumps(self.payload(analyzer.EXPERIMENT_ID, 1.0)), encoding="utf-8")
                repetition.write_text(json.dumps(self.payload(analyzer.EXPERIMENT_ID, 2.0)), encoding="utf-8")
                with patch.object(analyzer, "CURRENT_RUN_JSON", current), contextlib.redirect_stdout(io.StringIO()):
                    paths = analyzer._discover_run_files(runs_dir, include_current=True)
                    rows, run_index = analyzer._load_all_runs(paths)
                self.assertEqual(len(run_index), 2)
                self.assertEqual(sorted(rows["gpu_ms"].tolist()), [1.0, 2.0])
                summary = analyzer._build_run_summary(rows)
                self.assertEqual(sorted(summary["gpu_ms_median"].tolist()), [1.0, 2.0])

    def test_discovery_and_loading_count_current_archive_copy_only_once(self):
        for analyzer in self.analyzers:
            with self.subTest(experiment=analyzer.EXPERIMENT_ID):
                runs_dir = self.root / analyzer.EXPERIMENT_ID
                runs_dir.mkdir()
                current = runs_dir / "current.json"
                archive = runs_dir / "archive.json"
                payload = self.payload(analyzer.EXPERIMENT_ID, 1.0)
                current.write_text(json.dumps(payload), encoding="utf-8")
                archive.write_text(json.dumps(payload, indent=4), encoding="utf-8")
                with patch.object(analyzer, "CURRENT_RUN_JSON", current), contextlib.redirect_stdout(io.StringIO()):
                    paths = analyzer._discover_run_files(runs_dir, include_current=True)
                    rows, run_index = analyzer._load_all_runs(paths)
                self.assertEqual(paths, [current])
                self.assertEqual(len(run_index), 1)
                self.assertEqual(rows["gpu_ms"].tolist(), [1.0])


class ExperimentAnalysisLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzers = [(root, load_module(next((root / "scripts").glob("analyze_*.py"))))
                         for root in EXPERIMENT_ROOTS]
        cls.plotters = [(root, load_module(root / "scripts/plot_results.py")) for root in EXPERIMENT_ROOTS]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temporary_root = Path(self.temporary.name)

    @staticmethod
    def row(experiment_id):
        return {
            "experiment_id": experiment_id,
            "correctness_pass": True,
            "gpu_ms": 1.0,
            "end_to_end_ms": 2.0,
            "throughput": 1000.0,
            "gbps": 0.25,
        }

    def test_failed_or_missing_correctness_rejects_whole_run_before_analysis(self):
        source = self.temporary_root / "raw.json"
        for root, analyzer in self.analyzers:
            for flag in (False, None):
                with self.subTest(experiment=root.name, correctness=flag):
                    rows = [self.row(root.name), self.row(root.name) | {"correctness_pass": flag}]
                    source.write_text(json.dumps({"rows": rows}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "correctness"):
                        analyzer._load_frame(source)

    def test_nonfinite_measurements_reject_whole_run_before_ranking(self):
        source = self.temporary_root / "raw.json"
        for root, analyzer in self.analyzers:
            for metric in ("gpu_ms", "end_to_end_ms", "throughput", "gbps"):
                with self.subTest(experiment=root.name, metric=metric):
                    rows = [self.row(root.name), self.row(root.name) | {metric: math.nan}]
                    source.write_text(json.dumps({"rows": rows}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, metric):
                        analyzer._load_frame(source)

    def test_archive_fallback_filters_revision_before_modification_time(self):
        legacy = self.temporary_root / "legacy.json"
        fixed = self.temporary_root / "fixed.json"
        legacy.write_text(json.dumps({"metadata": {"exported_at_utc": "2026-04-01T12:00:00Z"},
                                      "rows": [self.row("example")]}))
        fixed.write_text(json.dumps({"metadata": {"measurement_revision": "fixed",
                                                "exported_at_utc": "2026-10-23T12:00:00Z"},
                                     "rows": [self.row("example")]}))
        os.utime(legacy, (1900000000, 1900000000))
        os.utime(fixed, (1800000000, 1800000000))
        for root, analyzer in self.analyzers:
            with self.subTest(experiment=root.name):
                with contextlib.redirect_stdout(io.StringIO()):
                    if inspect.signature(analyzer._latest_run_path).parameters:
                        actual = analyzer._latest_run_path(self.temporary_root)
                    else:
                        analyzer.RUNS_DIR = self.temporary_root
                        actual = analyzer._latest_run_path()
                self.assertEqual(actual, fixed)

    def test_incorrect_summary_never_reaches_plotting(self):
        import pandas as pd
        from analysis_integrity import write_analysis_provenance

        source = self.temporary_root / "summary.csv"
        pd.DataFrame([{"correctness_pass_rate": 0.5, "gpu_ms_median": 1.0}]).to_csv(source, index=False)
        raw = self.temporary_root / "raw.json"
        raw.write_text(json.dumps({"rows": [self.row("example")]}))
        with contextlib.redirect_stdout(io.StringIO()):
            write_analysis_provenance(self.temporary_root, [raw], "example", derived_tables=[source])
        for root, plotter in self.plotters:
            with self.subTest(experiment=root.name):
                if hasattr(plotter, "_load_table"):
                    loader, argument = plotter._load_table, source
                else:
                    plotter.TABLES_DIR = self.temporary_root
                    loader, argument = plotter._load, source.name
                with self.assertRaisesRegex(ValueError, "incorrect"):
                    loader(argument)

    def test_plot_entry_points_require_provenance_before_loading_tables(self):
        for root, plotter in self.plotters:
            with self.subTest(experiment=root.name):
                plotter.TABLES_DIR = self.temporary_root
                with self.assertRaisesRegex(ValueError, "provenance"):
                    plotter.main()


if __name__ == "__main__":
    unittest.main()
