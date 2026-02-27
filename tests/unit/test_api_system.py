"""Unit tests for expanded system/health endpoints (Task 3, Phase 4 Track B).

Uses TestClient (synchronous) for simplicity since new endpoints are simple.
"""

from __future__ import annotations

HTTP_200 = 200
HTTP_422 = 422


def test_health_includes_components() -> None:
    from fastapi.testclient import TestClient

    from finalayze.main import create_app

    client = TestClient(create_app())
    resp = client.get("/api/v1/health")
    assert resp.status_code == HTTP_200
    body = resp.json()
    assert "components" in body
    assert "db" in body["components"]
    assert "redis" in body["components"]


def test_health_feeds_returns_list() -> None:
    from fastapi.testclient import TestClient

    from finalayze.main import create_app

    client = TestClient(create_app())
    resp = client.get("/api/v1/health/feeds")
    assert resp.status_code == HTTP_200
    body = resp.json()
    assert "feeds" in body
    assert isinstance(body["feeds"], list)


def test_system_status_requires_api_key() -> None:
    from fastapi.testclient import TestClient

    from finalayze.main import create_app

    resp = TestClient(create_app()).get("/api/v1/system/status")
    assert resp.status_code == HTTP_422


def test_system_errors_returns_list() -> None:
    from config.settings import Settings
    from fastapi.testclient import TestClient

    from finalayze.main import create_app

    key = Settings().api_key
    resp = TestClient(create_app()).get("/api/v1/system/errors", headers={"X-API-Key": key})
    assert resp.status_code == HTTP_200
    assert isinstance(resp.json(), list)
