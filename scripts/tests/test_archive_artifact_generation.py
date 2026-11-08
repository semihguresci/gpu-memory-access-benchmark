from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_experiment_artifacts as artifacts


class ArchiveArtifactGenerationTests(unittest.TestCase):
    def test_advanced_experiments_analyze_and_plot_archives_without_current(self):
        for experiment_id in (
            "46_dependent_load_latency",
            "47_block_reorder_factorial",
            "48_buffer_vs_image_representation",
        ):
            with self.subTest(experiment=experiment_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runs = root / "runs"
                runs.mkdir()
                (runs / "archive.json").write_text('{"rows": []}', encoding="utf-8")
                original = artifacts.EXPERIMENT_CONFIG[experiment_id]
                config = replace(original, experiment_root=root, output_path=root / "results/tables/benchmark_results.json")
                with patch.dict(artifacts.EXPERIMENT_CONFIG, {experiment_id: config}), \
                     patch.object(artifacts, "run_command") as run:
                    self.assertTrue(artifacts.generate_experiment(experiment_id, collect_run=False))
                self.assertEqual(run.call_args_list, [
                    unittest.mock.call([sys.executable, str(config.analysis_script), "--skip-current"]),
                    unittest.mock.call([sys.executable, str(config.plot_script)]),
                ])


if __name__ == "__main__":
    unittest.main()
