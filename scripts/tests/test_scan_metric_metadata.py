from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ScanMetricMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "experiments/22_prefix_sum_scan/scripts/analyze_prefix_sum_scan.py"
        specification = importlib.util.spec_from_file_location("scan_metric_analyzer", path)
        cls.analyzer = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.analyzer)

    def test_raw_metric_model_survives_summary_without_pooling_different_models(self):
        basis = "estimated_source_level_storage_buffer_bytes"
        model = "source_storage_buffer_operations_not_physical_dram"
        rows = []
        for formula, byte_count in (("current_model", 20000), ("different_model", 10000)):
            for iteration in (0, 1):
                rows.append({
                    "experiment_id": "22_prefix_sum_scan", "variant": "items_per_thread_1",
                    "problem_size": 1024, "iteration": iteration, "dispatch_count": 3,
                    "gpu_ms": 1.0, "end_to_end_ms": 2.0, "gbps": byte_count / 1.0e6,
                    "throughput": 1024000.0, "correctness_pass": True,
                    "notes": f"gbps_basis={basis};traffic_model={model};estimated_traffic_formula={formula};"
                             f"estimated_global_total_bytes={byte_count}",
                })
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.json"
            source.write_text(json.dumps({"metadata": {}, "rows": rows}), encoding="utf-8")
            frame, _ = self.analyzer._load_frame(source)
        summary = self.analyzer._build_summary(frame)
        self.assertEqual(len(summary), 2)
        self.assertEqual(set(summary["gbps_basis"]), {basis})
        self.assertEqual(set(summary["traffic_model"]), {model})
        self.assertEqual(set(summary["estimated_traffic_formula"]), {"current_model", "different_model"})
        self.assertEqual(set(summary["estimated_global_total_bytes"]), {10000, 20000})
        self.assertEqual(summary["sample_count"].tolist(), [2, 2])
        relative = self.analyzer._build_relative(summary)
        self.assertEqual(len(relative), 2)
        self.assertEqual(relative["gbps_ratio_vs_baseline"].tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
