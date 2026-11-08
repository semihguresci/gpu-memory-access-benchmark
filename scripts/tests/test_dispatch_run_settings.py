"""Keep device comparisons within one validation and sampling configuration."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


class DispatchRunSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "experiments/01_dispatch_basics/scripts/analyze_dispatch_basics.py"
        specification = importlib.util.spec_from_file_location("dispatch_run_settings_test", path)
        cls.analyzer = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.analyzer)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.configurations = [(False, 2, 5, 1.0, 2.0), (True, 2, 5, 9.0, 3.0),
                               (False, 4, 5, 5.0, 4.0), (False, 2, 10, 7.0, 5.0)]
        self.paths = []
        for index, (validation, warmup, timed, latency, ratio) in enumerate(self.configurations):
            rows = []
            for size in (1024, 2048):
                for dispatch in (1, 64):
                    for variant in ("noop", "contiguous_write"):
                        gpu_ms = latency * size / 1024
                        if variant == "contiguous_write":
                            gpu_ms *= ratio
                        if dispatch == 64:
                            gpu_ms *= 0.5 if validation and variant == "contiguous_write" else 2.0
                        rows.append({
                            "experiment_id": "01_dispatch_basics", "variant": variant,
                            "problem_size": size, "dispatch_count": dispatch, "iteration": 0,
                            "gpu_ms": gpu_ms, "end_to_end_ms": gpu_ms + 1.0,
                            "throughput": size * 1000.0 / gpu_ms if variant != "noop" else 0.0,
                            "gbps": size * 4.0 / gpu_ms / 1.0e6 if variant != "noop" else 0.0,
                            "correctness_pass": True,
                        })
            payload = {"metadata": {
                "measurement_revision": "settings-test", "gpu_name": "same_gpu", "driver_version": "1",
                "vulkan_api_version": "1.2", "exported_at_utc": f"2026-11-08T12:00:0{index}Z",
                "validation_enabled": validation, "warmup_iterations": warmup, "timed_iterations": timed,
            }, "rows": rows}
            path = self.runs / f"run{index}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.paths.append(path)

    def device_summary(self):
        rows, _ = self.analyzer._load_all_runs(self.paths)
        return self.analyzer._build_device_summary(self.analyzer._build_run_summary(rows))

    def test_device_medians_keep_validation_warmup_and_timed_settings_separate(self):
        summary = self.device_summary()
        baseline = summary[(summary["variant"] == "noop") & (summary["problem_size"] == 1024)
                           & (summary["dispatch_count"] == 1)]
        self.assertEqual(sorted(baseline["gpu_ms_median"].tolist()), [1.0, 5.0, 7.0, 9.0])
        self.assertEqual(baseline["run_count"].tolist(), [1] * 4)
        self.assertEqual(len(baseline[["validation_enabled", "warmup_iterations", "timed_iterations"]]
                             .drop_duplicates()), 4)

    def test_main_preserves_settings_in_best_dispatch_ratios_and_chart_inputs(self):
        analyzer = self.analyzer
        tables = self.root / "results/tables"
        charts = self.root / "results/charts"
        plot_names = [name for name in vars(analyzer) if name.startswith("_plot_")]
        with contextlib.ExitStack() as stack:
            for name, value in {"ROOT": self.root, "TABLES_DIR": tables, "CHARTS_DIR": charts,
                                "CURRENT_RUN_JSON": tables / "benchmark_results.json"}.items():
                stack.enter_context(patch.object(analyzer, name, value))
            plots = {name: stack.enter_context(patch.object(analyzer, name, return_value=None))
                     for name in plot_names}
            stack.enter_context(patch("sys.argv", ["analyze", "--skip-current", "--runs-dir", str(self.runs)]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            analyzer.main()

        best = pd.read_csv(tables / "dispatch_basics_best_dispatch_by_device.csv")
        writes = best[best["variant"] == "contiguous_write"]
        self.assertEqual(len(writes), 8)
        self.assertEqual(set(writes.loc[writes["validation_enabled"], "best_dispatch_for_latency"]), {64})
        self.assertEqual(set(writes.loc[~writes["validation_enabled"], "best_dispatch_for_latency"]), {1})
        ratios = pd.read_csv(tables / "dispatch_basics_operation_ratio_by_device.csv")
        self.assertEqual(len(ratios), 16)
        for validation, warmup, timed, _, expected_ratio in self.configurations:
            subset = ratios[(ratios["validation_enabled"] == validation)
                            & (ratios["warmup_iterations"] == warmup) & (ratios["timed_iterations"] == timed)
                            & (ratios["dispatch_count"] == 1)]
            self.assertEqual(subset["gpu_ms_ratio_write_over_noop"].tolist(), [expected_ratio] * 2)
        index = pd.read_csv(tables / "dispatch_basics_device_chart_index.csv")
        self.assertEqual(len(index), 4)
        self.assertEqual(index["chart"].nunique(), 4)
        per_device_calls = plots["_plot_time_throughput"].call_args_list[1:]
        self.assertEqual(len(per_device_calls), 4)
        for call in per_device_calls:
            frame = call.args[0]
            self.assertEqual(len(frame[["validation_enabled", "warmup_iterations", "timed_iterations"]]
                                 .drop_duplicates()), 1)

    def test_comparison_plots_draw_distinct_labeled_curves_for_each_configuration(self):
        analyzer = self.analyzer
        summary = self.device_summary()
        ratios = analyzer._build_operation_ratio_table(summary)
        for plot, data in ((analyzer._plot_cross_device_comparison, summary),
                           (analyzer._plot_operation_ratio_by_device, ratios)):
            with self.subTest(plot=plot.__name__), patch("matplotlib.figure.Figure.savefig"), \
                    patch.object(analyzer.plt, "close"):
                plot(data, self.root / "chart.png")
                figure = analyzer.plt.gcf()
                lines = figure.axes[0].get_lines()
                self.assertEqual(len(lines), 4)
                self.assertEqual(len({line.get_label() for line in lines}), 4)
                self.assertTrue(all(len(line.get_xdata()) == 2 for line in lines))
                self.assertTrue(all("validation " in line.get_label() and "warmup " in line.get_label()
                                    and "timed " in line.get_label() for line in lines))
            analyzer.plt.close(figure)

    def test_single_configuration_preserves_existing_device_chart_filename(self):
        summary = self.device_summary()
        summary = summary[(summary["validation_enabled"] == False) & (summary["warmup_iterations"] == 2)
                          & (summary["timed_iterations"] == 5)]
        index = self.analyzer._build_chart_index(summary)
        self.assertEqual(index["chart"].tolist(), [
            "results/charts/dispatch_basics_same_gpu_1_1_2_time_throughput.png",
        ])


if __name__ == "__main__":
    unittest.main()
