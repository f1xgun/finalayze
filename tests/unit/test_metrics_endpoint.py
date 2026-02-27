"""Integration smoke test: /metrics endpoint is accessible without auth."""

from __future__ import annotations

from fastapi.testclient import TestClient

from finalayze.main import create_app


def test_metrics_endpoint_no_auth_required() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "finalayze" in resp.text or "python_gc" in resp.text  # prometheus output


def test_metrics_endpoint_not_in_openapi_schema() -> None:
    app = create_app()
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    paths = schema.get("paths", {})
    assert "/metrics" not in paths
