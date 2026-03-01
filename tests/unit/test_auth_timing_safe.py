"""Tests for 6D.7: timing-safe API key comparison via hmac.compare_digest."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from finalayze.api.v1.auth import api_key_auth, require_api_key

_TEST_KEY = "test-secret-key-12345"


@pytest.fixture
def app_with_require() -> FastAPI:
    """App using require_api_key factory."""
    app = FastAPI()

    @app.get("/test", dependencies=[Depends(require_api_key(_TEST_KEY))])
    async def _endpoint() -> dict[str, str]:
        return {"ok": "true"}

    return app


@pytest.fixture
def app_with_runtime_auth() -> FastAPI:
    """App using api_key_auth (settings-based)."""
    app = FastAPI()

    @app.get("/test", dependencies=[Depends(api_key_auth)])
    async def _endpoint() -> dict[str, str]:
        return {"ok": "true"}

    return app


def test_require_api_key_uses_hmac_compare_digest(app_with_require: FastAPI) -> None:
    """require_api_key must use hmac.compare_digest, not ==."""
    client = TestClient(app_with_require)
    with patch("finalayze.api.v1.auth.hmac.compare_digest", return_value=True) as mock_cmp:
        resp = client.get("/test", headers={"X-API-Key": _TEST_KEY})
        assert resp.status_code == 200
        mock_cmp.assert_called_once_with(_TEST_KEY, _TEST_KEY)


def test_api_key_auth_uses_hmac_compare_digest(app_with_runtime_auth: FastAPI) -> None:
    """api_key_auth must use hmac.compare_digest, not ==."""
    client = TestClient(app_with_runtime_auth)
    with (
        patch("finalayze.api.v1.auth.hmac.compare_digest", return_value=True) as mock_cmp,
        patch("config.settings.get_settings") as mock_settings,
    ):
        mock_settings.return_value.api_key = _TEST_KEY
        resp = client.get("/test", headers={"X-API-Key": _TEST_KEY})
        assert resp.status_code == 200
        mock_cmp.assert_called_once_with(_TEST_KEY, _TEST_KEY)


def test_require_api_key_rejects_wrong_key(app_with_require: FastAPI) -> None:
    """Wrong key should still be rejected (hmac.compare_digest returns False)."""
    client = TestClient(app_with_require)
    resp = client.get("/test", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401
