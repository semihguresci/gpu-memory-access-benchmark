#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "config" / "experiment_manifest.json"


def load_experiment_manifest(manifest_path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    # Python tooling reads the same JSON manifest as CMake so experiment registration
    # cannot drift between build, collection, and artifact-generation paths.
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError(f"Experiment manifest at {manifest_path} is missing an 'experiments' list.")

    return experiments


def enabled_experiment_ids() -> list[str]:
    return [str(experiment["id"]) for experiment in load_experiment_manifest() if bool(experiment["enabled"])]
