# tests/unit/test_api_auth.py
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.testclient import TestClient

from finalayze.api.v1.auth import require_api_key

if TYPE_CHECKING:
    import pytest


def _make_app(key: str) -> FastAPI:
    from fastapi import Depends

    app = FastAPI()

    @app.get("/secret")
    async def secret(_: None = Depends(require_api_key(key))) -> dict[str, str]:
        return {"ok": "yes"}

    return app


def test_valid_key_passes() -> None:
    client = TestClient(_make_app("test-key"))
    resp = client.get("/secret", headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200


def test_wrong_key_returns_401() -> None:
    client = TestClient(_make_app("test-key"))
    resp = client.get("/secret", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_missing_key_returns_422() -> None:
    client = TestClient(_make_app("test-key"))
    resp = client.get("/secret")
    assert resp.status_code == 422


def test_key_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """API key must never appear in logs."""
    import logging

    with caplog.at_level(logging.DEBUG):
        client = TestClient(_make_app("super-secret-key"))
        client.get("/secret", headers={"X-API-Key": "super-secret-key"})
    assert "super-secret-key" not in caplog.text
