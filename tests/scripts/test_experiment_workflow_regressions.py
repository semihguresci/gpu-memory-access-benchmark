from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import os
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_integrity import latest_compatible_run
from experiment_gui import ExperimentRunnerGui


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location(
        "workflow_regression_" + path.parent.parent.name + "_" + path.stem, path
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ArchiveWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_run(self, name, timestamp, gpu_ms=1.0):
        path = self.root / name
        path.write_text(json.dumps({
            "metadata": {"measurement_revision": "fixed", "exported_at_utc": timestamp},
            "rows": [{"experiment_id": "example", "gpu_ms": gpu_ms}],
        }), encoding="utf-8")
        return path

    def test_every_collector_accepts_an_external_archive_directory(self):
        input_path = self.root / "input.json"
        for script in sorted((ROOT / "experiments").glob("*/scripts/collect_run.py")):
            collector = load_module(script)
            with self.subTest(experiment=collector.EXPERIMENT_ID):
                destination = self.root / collector.EXPERIMENT_ID
                payload = {
                    "metadata": {"gpu_name": "test_gpu", "exported_at_utc": "2026-11-08T12:00:00Z"},
                    "rows": [{"experiment_id": collector.EXPERIMENT_ID}],
                }
                input_path.write_text(json.dumps(payload), encoding="utf-8")
                with patch.object(sys, "argv", [str(script), "--input", str(input_path),
                                                "--runs-dir", str(destination)]), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    collector.main()
                archives = list(destination.rglob("*.json"))
                self.assertEqual(len(archives), 1)
                self.assertEqual(json.loads(archives[0].read_text(encoding="utf-8")), payload)
                self.assertIn(archives[0].resolve().as_posix(), output.getvalue().replace("\\", "/"))

    def test_every_single_archive_selector_uses_export_time_within_revision(self):
        older = self.write_run("older.json", "2026-11-07T12:00:00Z")
        newer = self.write_run("newer.json", "2026-11-08T12:00:00Z", 2.0)
        os.utime(older, (1900000000, 1900000000))
        os.utime(newer, (1800000000, 1800000000))
        checked = 0
        for script in sorted((ROOT / "experiments").glob("*/scripts/analyze_*.py")):
            analyzer = load_module(script)
            if not hasattr(analyzer, "_latest_run_path"):
                continue
            checked += 1
            with self.subTest(experiment=analyzer.EXPERIMENT_ID), \
                    contextlib.redirect_stdout(io.StringIO()):
                if inspect.signature(analyzer._latest_run_path).parameters:
                    actual = analyzer._latest_run_path(self.root)
                else:
                    with patch.object(analyzer, "RUNS_DIR", self.root):
                        actual = analyzer._latest_run_path()
                self.assertEqual(actual, newer)
        self.assertGreater(checked, 0)

    def test_archive_timestamp_falls_back_to_mtime_when_missing_or_invalid(self):
        older = self.write_run("older.json", "")
        newer = self.write_run("newer.json", "not-a-timestamp", 2.0)
        os.utime(older, (1700000000, 1700000000))
        os.utime(newer, (1800000000, 1800000000))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(latest_compatible_run([older, newer]), newer)


class ArtifactGuiWorkflowTests(unittest.TestCase):
    def make_gui(self):
        gui = object.__new__(ExperimentRunnerGui)
        gui.worker_thread = None
        gui.stop_requested = threading.Event()
        gui.event_queue = queue.Queue()
        gui.binary_var = Mock()
        gui.status_var = Mock()
        gui._set_busy_state = Mock()
        gui._selected_experiment_ids = Mock(return_value=["example"])
        gui.configs = {"example": SimpleNamespace(id="example", display_name="Example")}
        return gui

    def test_artifacts_start_ignores_benchmark_iteration_settings(self):
        gui = self.make_gui()
        gui._validate_run_settings = Mock(side_effect=AssertionError("benchmark settings read"))
        with patch("experiment_gui.threading.Thread") as thread:
            gui._start_experiment_worker("artifacts")
        gui._validate_run_settings.assert_not_called()
        thread.return_value.start.assert_called_once()

    def test_artifacts_run_without_a_binary(self):
        gui = self.make_gui()
        gui._build_artifact_command = Mock(return_value=["generate-artifacts"])
        gui._run_streaming_command = Mock()
        with patch("experiment_gui.resolve_binary", side_effect=FileNotFoundError("missing binary")) as resolve:
            gui._run_experiment_worker("artifacts", ["example"])
        resolve.assert_not_called()
        gui.binary_var.get.assert_not_called()
        gui._run_streaming_command.assert_called_once_with(["generate-artifacts"])
        events = list(gui.event_queue.queue)
        self.assertEqual(events[-1], ("finished", True, "Completed artifacts for 1 experiment(s)."))

    def test_benchmark_run_still_requires_valid_settings_and_binary(self):
        gui = self.make_gui()
        gui._validate_run_settings = Mock(return_value=False)
        with patch("experiment_gui.threading.Thread") as thread:
            gui._start_experiment_worker("run")
        gui._validate_run_settings.assert_called_once()
        thread.assert_not_called()
        gui.binary_var.get.return_value = "missing.exe"
        with patch("experiment_gui.resolve_binary", side_effect=FileNotFoundError("missing binary")):
            gui._run_experiment_worker("run", ["example"])
        self.assertEqual(list(gui.event_queue.queue)[-1], ("finished", False, "missing binary"))


class LocalSizeChartLabelTests(unittest.TestCase):
    def test_both_plot_paths_keep_nonpower_sizes_distinct(self):
        import pandas as pd
        from matplotlib.axes import Axes

        experiment = ROOT / "experiments/02_local_size_sweep/scripts"
        analyzer = load_module(experiment / "analyze_local_size_sweep.py")
        plotter = load_module(experiment / "plot_results.py")
        rows = [{
            "experiment_id": "02_local_size_sweep", "variant": "contiguous_write_ls64",
            "base_variant": "contiguous_write", "problem_size": size, "local_size_x": 64,
            "dispatch_count": 1, "gpu_ms": 1.0, "gpu_ms_median": 1.0,
            "throughput": 1000.0, "throughput_median": 1000.0, "end_to_end_ms": 2.0,
            "gbps": 0.25, "correctness_pass": True,
        } for size in (2097152, 2796202)]
        original_plot = Axes.plot
        labels = []

        def record_plot(axis, *args, **kwargs):
            if "label" in kwargs:
                labels.append(kwargs["label"])
            return original_plot(axis, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            raw = output / "raw.json"
            raw.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            with patch.object(Axes, "plot", record_plot):
                analyzer._plot_metric_vs_local_size(pd.DataFrame(rows), "gpu_ms_median", "GPU ms",
                                                    output / "analyzer.png", "Example")
            self.assertEqual(set(labels), {"d=1, size=2^21", "d=1, size=2,796,202"})
            labels.clear()
            with patch.object(plotter, "INPUT_JSON", raw), patch.object(plotter, "OUTPUT_PNG", output / "plot.png"), \
                    patch.object(Axes, "plot", record_plot):
                self.assertTrue(plotter._plot_local_size_sweep_summary())
            self.assertEqual(set(labels), {"d=1, size=2^21", "d=1, size=2,796,202"})


if __name__ == "__main__":
    unittest.main()
