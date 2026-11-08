"""Where Theria keeps its files: /data in the add-on, THERIA_DATA_DIR for a local run.

A local run (dev-start.sh) copies the add-on's settings and state into its own directory first, so it
plans from the same settings, models and memory as the add-on.
"""

import json
import os
from collections.abc import Mapping
from typing import Any

# The files a plan depends on. History files (records, runs, traces, logs) stay where they are.
DATA_FILES = (
    "options.json",
    "plan_overrides.json",
    "plan_state.json",
    "radiator_prefs.json",
)
SECRET_SECTIONS = ("influxdb",)  # a local run reads its InfluxDB login from .env


def data_dir(environ: Mapping[str, str]) -> str:
    return environ.get("THERIA_DATA_DIR", "/data")


def read_data_files(directory: str) -> dict[str, Any]:
    """The settings and state files that exist, by name, without secrets."""
    files: dict[str, Any] = {}
    for name in DATA_FILES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            files[name] = json.load(f)
    if "options.json" in files:
        files["options.json"] = {
            k: v for k, v in files["options.json"].items() if k not in SECRET_SECTIONS
        }
    return files


def write_data_files(directory: str, files: Mapping[str, Any]) -> None:
    """Make the settings and state files in `directory` what `files` says; a missing one is removed."""
    os.makedirs(directory, exist_ok=True)
    for name in DATA_FILES:
        path = os.path.join(directory, name)
        if name in files:
            with open(path, "w") as f:
                json.dump(files[name], f, indent=1)
        elif os.path.exists(path):
            os.remove(path)
