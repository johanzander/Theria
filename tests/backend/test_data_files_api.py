"""The add-on hands its settings and state to a local run (dev-start.sh) through ingress."""

import json
import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

import api

import app as backend_app


def test_the_settings_and_state_files_are_handed_out_without_secrets(
    tmp_path, monkeypatch
):
    (tmp_path / "plan_overrides.json").write_text(json.dumps({"target_c": 18.5}))
    (tmp_path / "options.json").write_text(
        json.dumps({"plan": {}, "influxdb": {"password": "secret"}})
    )
    monkeypatch.setattr(api, "DATA_DIR", str(tmp_path))
    body = TestClient(backend_app.app).get("/api/data-files").json()
    assert body == {
        "options.json": {"plan": {}},
        "plan_overrides.json": {"target_c": 18.5},
    }
