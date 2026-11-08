from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("cmake"), "CMake is required for the regeneration test")
class ManifestDependencyTests(unittest.TestCase):
    def test_manifest_only_edit_regenerates_registry_inputs_on_build(self) -> None:
        with tempfile.TemporaryDirectory(prefix="manifest_dependency_") as directory:
            source = Path(directory)
            build = source / "build"
            manifest = source / "config" / "experiment_manifest.json"
            manifest.parent.mkdir()
            entry = {
                "id": "test_experiment",
                "display_name": "Test experiment",
                "category": "test",
                "adapter_symbol": "run_test_adapter",
                "enabled": True,
                "source": "test.cpp",
                "adapter_source": "adapter.cpp",
                "required_local_size": [32, 1, 1],
                "required_shared_memory_bytes": 0,
            }
            manifest.write_text(json.dumps({"experiments": [entry]}), encoding="utf-8")
            loader = (REPO_ROOT / "cmake" / "experiments_manifest.cmake").as_posix()
            (source / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.21)\n'
                'project(manifest_dependency_test NONE)\n'
                f'include("{loader}")\n'
                'configure_file(manifest.txt.in manifest.txt @ONLY)\n'
                'add_custom_target(manifest_probe ALL COMMAND "${CMAKE_COMMAND}" -E true)\n',
                encoding="utf-8",
            )
            (source / "manifest.txt.in").write_text("@EXPERIMENT_MANIFEST_ENTRIES@", encoding="utf-8")
            configure = ["cmake", "-S", str(source), "-B", str(build)]
            if shutil.which("ninja"):
                configure.extend(["-G", "Ninja"])
            self._run(configure)
            generated = build / "manifest.txt"
            self.assertIn("|ON|32|1|1|0", generated.read_text(encoding="utf-8"))

            entry["enabled"] = False
            entry["required_local_size"] = [64, 1, 1]
            entry["required_shared_memory_bytes"] = 2048
            manifest.write_text(json.dumps({"experiments": [entry]}), encoding="utf-8")
            # Ensure the dependency is newer even on filesystems with coarse timestamps.
            modified = max(time.time(), generated.stat().st_mtime + 2.0)
            os.utime(manifest, (modified, modified))
            self._run(["cmake", "--build", str(build)])
            self.assertIn("|OFF|64|1|1|2048", generated.read_text(encoding="utf-8"))

    def _run(self, command: list[str]) -> None:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
