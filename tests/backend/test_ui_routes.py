"""The new React page is the start page; the old dashboard lives at /legacy."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

import app as backend_app

INGRESS = "/api/hassio_ingress/abc123"


@pytest.fixture
def client(tmp_path, monkeypatch):
    built = tmp_path / "plan"
    built.mkdir()
    (built / "index.html").write_text(
        '<!doctype html><html><head><title>Theria - Price and plan</title></head>'
        '<body><div id="root"></div></body></html>'
    )
    monkeypatch.setattr(backend_app, "plan_ui_dir", str(built))
    return TestClient(backend_app.app)


def test_the_start_page_is_the_new_app_with_assets_found_behind_ingress(client):
    html = client.get("/", headers={"X-Ingress-Path": INGRESS}).text
    assert 'id="root"' in html
    assert f'<base href="{INGRESS}/static/plan/">' in html


def test_without_ingress_the_base_is_the_plain_static_path(client):
    assert '<base href="/static/plan/">' in client.get("/").text


def test_the_old_dashboard_is_still_available_at_legacy(client):
    html = client.get("/legacy", headers={"X-Ingress-Path": INGRESS}).text
    assert 'id="root"' not in html and "THERIA" in html.upper()
    assert f'<base href="{INGRESS}/">' in html


def test_an_unbuilt_frontend_says_so_instead_of_showing_the_old_page(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(backend_app, "plan_ui_dir", str(tmp_path / "missing"))
    response = client.get("/")
    assert response.status_code == 503 and "npm run build" in response.text
