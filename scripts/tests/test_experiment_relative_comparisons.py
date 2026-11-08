from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_analyzer(experiment: str):
    path = next((ROOT / "experiments" / experiment / "scripts").glob("analyze_*.py"))
    specification = importlib.util.spec_from_file_location("relative_test_" + experiment, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class BlockReorderRelativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = load_analyzer("47_block_reorder_factorial")

    def test_variants_use_sequential_ordered_baseline_at_matching_width_and_size(self):
        rows = []
        modes = (("seq", "ordered", 1.0), ("seq", "reversed", 1.5),
                 ("shuffled", "ordered", 2.0), ("shuffled", "reversed", 1.81))
        for size in (1024, 4096):
            for width in (64, 256):
                baseline_ms = size / width
                for block_mode, within_mode, ratio in modes:
                    rows.append({
                        "variant": f"w{width}_blocks_{block_mode}_within_{within_mode}",
                        "problem_size": size, "block_width": width, "residency": "warm",
                        "gpu_ms_median": baseline_ms * ratio,
                        "throughput_median": 100.0 / ratio, "gbps_median": 10.0 / ratio,
                    })
                rows.append(rows[-1] | {"variant": rows[-1]["variant"] + "_evicted",
                                        "residency": "evicted_buffer_sweep", "gpu_ms_median": 999.0})
        relative = self.analyzer._build_relative(pd.DataFrame(rows))
        self.assertEqual(len(relative), 16)
        for row in relative.itertuples():
            self.assertEqual(row.baseline_variant, f"w{row.block_width}_blocks_seq_within_ordered")
            expected = next(ratio for block_mode, within_mode, ratio in modes
                            if row.variant.endswith(f"blocks_{block_mode}_within_{within_mode}"))
            self.assertAlmostEqual(row.slowdown_vs_same_width_baseline, expected)
            self.assertAlmostEqual(row.gbps_ratio_vs_baseline, 1.0 / expected)

    def test_missing_matching_width_baseline_is_rejected(self):
        rows = [{
            "variant": "w128_blocks_shuffled_within_ordered", "problem_size": 1024,
            "block_width": 128, "residency": "warm", "gpu_ms_median": 2.0,
            "throughput_median": 100.0, "gbps_median": 10.0,
        }]
        with self.assertRaisesRegex(ValueError, "baseline missing"):
            self.analyzer._build_relative(pd.DataFrame(rows))


class HeapPlacementRelativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = load_analyzer("28_device_local_vs_host_visible_heap_placement")

    @staticmethod
    def summary():
        rows = []
        for size in (1024, 4096):
            for kernels, direct_ms, staged_ms in ((1, 2.0, 4.0), (8, 20.0, 10.0)):
                for placement, gpu_ms in (("host_visible_direct", direct_ms),
                                          ("device_local_staged", staged_ms)):
                    rows.append({
                        "variant": placement + (f"_reuse_k{kernels}" if kernels > 1 else ""),
                        "placement": placement, "problem_size": size, "dispatch_count": kernels,
                        "chain_kernel_count": kernels, "gpu_ms_median": gpu_ms * size,
                        "end_to_end_ms_median": gpu_ms * size * 2.0,
                        "gbps_median": kernels / gpu_ms, "upload_ms_median": 0.0,
                        "readback_ms_median": 0.0,
                    })
        return pd.DataFrame(rows)

    def test_reuse_variants_compare_with_direct_at_the_same_kernel_count(self):
        relative = self.analyzer._build_relative(self.summary())
        self.assertEqual(len(relative), 8)
        for row in relative.itertuples():
            expected_baseline = "host_visible_direct" + (
                f"_reuse_k{row.chain_kernel_count}" if row.chain_kernel_count > 1 else ""
            )
            self.assertEqual(row.baseline_variant, expected_baseline)
            self.assertEqual(row.dispatch_count, row.chain_kernel_count)
            expected = 1.0 if row.variant.startswith("host_visible_direct") else (
                2.0 if row.chain_kernel_count == 1 else 0.5
            )
            self.assertAlmostEqual(row.slowdown_vs_host_visible_gpu_path, expected)
            self.assertAlmostEqual(row.slowdown_vs_host_visible_harness_wall_clock, expected)
            self.assertAlmostEqual(row.gpu_path_gbps_ratio_vs_host_visible, 1.0 / expected)

    def test_missing_reuse_baseline_does_not_fall_back_to_one_kernel(self):
        summary = self.summary()
        summary = summary[summary["variant"] != "host_visible_direct_reuse_k8"]
        with self.assertRaisesRegex(ValueError, "baseline missing"):
            self.analyzer._build_relative(summary)

    def test_mismatched_dispatch_count_does_not_match_by_chain_count_alone(self):
        summary = self.summary()
        summary.loc[summary["variant"] == "device_local_staged_reuse_k8", "dispatch_count"] = 9
        with self.assertRaisesRegex(ValueError, "baseline missing"):
            self.analyzer._build_relative(summary)

    def test_legacy_single_kernel_rows_without_reuse_note_remain_supported(self):
        rows = [{
            "experiment_id": self.analyzer.EXPERIMENT_ID, "variant": placement,
            "problem_size": 1024, "dispatch_count": 1, "iteration": 0,
            "correctness_pass": True, "gpu_ms": gpu_ms, "end_to_end_ms": gpu_ms * 2.0,
            "throughput": 1024.0 / gpu_ms, "gbps": 8.0 / gpu_ms,
            "notes": f"placement={placement}",
        } for placement, gpu_ms in (("host_visible_direct", 1.0), ("device_local_staged", 2.0))]
        with patch.object(Path, "read_text", return_value=json.dumps({"rows": rows})):
            frame, _ = self.analyzer._load_frame(Path("legacy.json"))
        relative = self.analyzer._build_relative(self.analyzer._build_summary(frame))
        self.assertEqual(relative["chain_kernel_count"].tolist(), [1, 1])
        staged = relative[relative["variant"] == "device_local_staged"].iloc[0]
        self.assertEqual(staged["slowdown_vs_host_visible_gpu_path"], 2.0)


if __name__ == "__main__":
    unittest.main()
