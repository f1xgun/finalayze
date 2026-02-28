"""Unit tests for the Finalayze FastAPI health and mode endpoints.

All tests use httpx.AsyncClient with ASGITransport -- no live server needed.
Each test class uses FastAPI's dependency_overrides to inject a fresh
ModeManager, ensuring full isolation between tests.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from config.settings import Settings, get_settings
from httpx import ASGITransport, AsyncClient

from finalayze.api.v1.system import ModeManager, get_mode_manager
from finalayze.main import create_app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENV_VAR_REAL_CONFIRMED = "FINALAYZE_REAL_CONFIRMED"
ENV_VAR_REAL_TOKEN = "FINALAYZE_REAL_TOKEN"  # noqa: S105
API_VERSION = "0.1.0"

STATUS_OK = "ok"

MODE_DEBUG = "debug"
MODE_SANDBOX = "sandbox"
MODE_REAL = "real"
MODE_INVALID = "invalid_mode"

HTTP_200 = 200
HTTP_400 = 400
HTTP_403 = 403
HTTP_422 = 422


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_test_app() -> tuple[object, ModeManager]:
    """Create a fresh FastAPI app with an isolated ModeManager for testing."""
    fresh_manager = ModeManager()
    application = create_app()
    application.dependency_overrides[get_mode_manager] = lambda: fresh_manager  # type: ignore[attr-defined]
    return application, fresh_manager


def _patch_liveness_ok() -> tuple[object, object]:
    """Return context managers that patch DB and Redis health checks to return 'ok'."""
    db_patch = patch("finalayze.api.v1.system._check_db", new_callable=AsyncMock, return_value="ok")
    redis_patch = patch(
        "finalayze.api.v1.system._check_redis", new_callable=AsyncMock, return_value="ok"
    )
    return db_patch, redis_patch


def make_client(application: object) -> AsyncClient:
    """Create an async test client for the given ASGI app."""
    return AsyncClient(transport=ASGITransport(app=application), base_url="http://test")  # type: ignore[arg-type]


def get_api_key() -> str:
    return Settings().api_key


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self) -> None:
        app, _ = build_test_app()
        async with make_client(app) as client:
            response = await client.get("/api/v1/health")
        assert response.status_code == HTTP_200

    @pytest.mark.asyncio
    async def test_health_status_is_ok(self) -> None:
        app, _ = build_test_app()
        db_patch, redis_patch = _patch_liveness_ok()
        with db_patch, redis_patch:
            async with make_client(app) as client:
                response = await client.get("/api/v1/health")
        assert response.json()["status"] == STATUS_OK

    @pytest.mark.asyncio
    async def test_health_status_is_degraded_when_db_down(self) -> None:
        app, _ = build_test_app()
        with (
            patch(
                "finalayze.api.v1.system._check_db", new_callable=AsyncMock, return_value="error"
            ),
            patch(
                "finalayze.api.v1.system._check_redis", new_callable=AsyncMock, return_value="ok"
            ),
        ):
            async with make_client(app) as client:
                response = await client.get("/api/v1/health")
        assert response.json()["status"] == "degraded"
        assert response.json()["components"]["db"] == "error"

    @pytest.mark.asyncio
    async def test_health_components_structure(self) -> None:
        app, _ = build_test_app()
        db_patch, redis_patch = _patch_liveness_ok()
        with db_patch, redis_patch:
            async with make_client(app) as client:
                response = await client.get("/api/v1/health")
        components = response.json()["components"]
        assert "db" in components
        assert "redis" in components

    @pytest.mark.asyncio
    async def test_health_mode_is_debug_by_default(self) -> None:
        app, _ = build_test_app()
        async with make_client(app) as client:
            response = await client.get("/api/v1/health")
        assert response.json()["mode"] == MODE_DEBUG

    @pytest.mark.asyncio
    async def test_health_version_present(self) -> None:
        app, _ = build_test_app()
        async with make_client(app) as client:
            response = await client.get("/api/v1/health")
        assert response.json()["version"] == API_VERSION


# ---------------------------------------------------------------------------
# GET /api/v1/mode
# ---------------------------------------------------------------------------
class TestGetModeEndpoint:
    @pytest.mark.asyncio
    async def test_get_mode_returns_200(self) -> None:
        app, _ = build_test_app()
        key = get_api_key()
        async with make_client(app) as client:
            response = await client.get("/api/v1/mode", headers={"X-API-Key": key})
        assert response.status_code == HTTP_200

    @pytest.mark.asyncio
    async def test_get_mode_returns_debug_by_default(self) -> None:
        app, _ = build_test_app()
        key = get_api_key()
        async with make_client(app) as client:
            response = await client.get("/api/v1/mode", headers={"X-API-Key": key})
        assert response.json()["mode"] == MODE_DEBUG


# ---------------------------------------------------------------------------
# POST /api/v1/mode
# ---------------------------------------------------------------------------
class TestSetModeEndpoint:
    @pytest.mark.asyncio
    async def test_set_mode_to_sandbox_returns_200(self) -> None:
        app, _ = build_test_app()
        key = get_api_key()
        async with make_client(app) as client:
            response = await client.post(
                "/api/v1/mode", json={"mode": MODE_SANDBOX}, headers={"X-API-Key": key}
            )
        assert response.status_code == HTTP_200

    @pytest.mark.asyncio
    async def test_set_mode_to_sandbox_changes_mode(self) -> None:
        app, _ = build_test_app()
        key = get_api_key()
        async with make_client(app) as client:
            await client.post(
                "/api/v1/mode", json={"mode": MODE_SANDBOX}, headers={"X-API-Key": key}
            )
            response = await client.get("/api/v1/mode", headers={"X-API-Key": key})
        assert response.json()["mode"] == MODE_SANDBOX

    @pytest.mark.asyncio
    async def test_set_mode_response_contains_new_mode(self) -> None:
        app, _ = build_test_app()
        key = get_api_key()
        async with make_client(app) as client:
            response = await client.post(
                "/api/v1/mode", json={"mode": MODE_SANDBOX}, headers={"X-API-Key": key}
            )
        assert response.json()["mode"] == MODE_SANDBOX

    @pytest.mark.asyncio
    async def test_set_mode_to_real_without_token_returns_403(self) -> None:
        """When real_token is not configured, transitioning to REAL must be denied with 403."""
        os.environ.pop(ENV_VAR_REAL_TOKEN, None)
        os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)
        app, _ = build_test_app()
        key = get_api_key()
        async with make_client(app) as client:
            response = await client.post(
                "/api/v1/mode", json={"mode": MODE_REAL}, headers={"X-API-Key": key}
            )
        assert response.status_code == HTTP_403

    @pytest.mark.asyncio
    async def test_set_mode_to_real_with_token_and_env_returns_200(self) -> None:
        """When real_token is set and confirm_token matches, and env confirms, return 200."""
        secret = "my-secret-token"  # noqa: S105
        os.environ[ENV_VAR_REAL_TOKEN] = secret
        os.environ[ENV_VAR_REAL_CONFIRMED] = "true"
        get_settings.cache_clear()
        try:
            app, _ = build_test_app()
            key = get_api_key()
            async with make_client(app) as client:
                response = await client.post(
                    "/api/v1/mode",
                    json={"mode": MODE_REAL, "confirm_token": secret},
                    headers={"X-API-Key": key},
                )
            assert response.status_code == HTTP_200
        finally:
            os.environ.pop(ENV_VAR_REAL_TOKEN, None)
            os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_set_mode_to_real_error_detail_present(self) -> None:
        os.environ.pop(ENV_VAR_REAL_TOKEN, None)
        os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)
        app, _ = build_test_app()
        key = get_api_key()
        async with make_client(app) as client:
            response = await client.post(
                "/api/v1/mode", json={"mode": MODE_REAL}, headers={"X-API-Key": key}
            )
        assert "detail" in response.json()

    @pytest.mark.asyncio
    async def test_set_invalid_mode_returns_422(self) -> None:
        app, _ = build_test_app()
        key = get_api_key()
        async with make_client(app) as client:
            response = await client.post(
                "/api/v1/mode", json={"mode": MODE_INVALID}, headers={"X-API-Key": key}
            )
        assert response.status_code == HTTP_422
