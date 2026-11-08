from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from run_experiment_data_collection import SIZE_PRESETS, build_run_label, resolve_scratch_size


class ScratchSizeResolutionTests(unittest.TestCase):
    def test_omitted_size_and_preset_uses_experiment_default(self) -> None:
        self.assertIsNone(resolve_scratch_size(None, None))

    def test_explicit_size_is_forwarded_unchanged(self) -> None:
        self.assertEqual(resolve_scratch_size("4M", None), "4M")
        self.assertEqual(resolve_scratch_size("256M", None), "256M")

    def test_preset_expands_to_mapped_size(self) -> None:
        for preset, expected in SIZE_PRESETS.items():
            with self.subTest(preset=preset):
                self.assertEqual(resolve_scratch_size(None, preset), expected)

    def test_size_and_preset_cannot_be_combined(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_scratch_size("4M", "small")

    def test_unknown_preset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown --size-preset"):
            resolve_scratch_size(None, "huge")


class RunLabelTests(unittest.TestCase):
    def test_single_run_keeps_label_unchanged(self) -> None:
        self.assertEqual(build_run_label("", 1, 1), "")
        self.assertEqual(build_run_label("integrity_fixes", 1, 1), "integrity_fixes")

    def test_multiple_runs_append_index(self) -> None:
        self.assertEqual(build_run_label("", 1, 3), "run1")
        self.assertEqual(build_run_label("", 3, 3), "run3")
        self.assertEqual(build_run_label("repeat", 2, 3), "repeat_run2")


if __name__ == "__main__":
    unittest.main()
