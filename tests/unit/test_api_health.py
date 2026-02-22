"""Unit tests for the Finalayze FastAPI health and mode endpoints.

All tests use httpx.AsyncClient with ASGITransport -- no live server needed.
Each test class uses FastAPI's dependency_overrides to inject a fresh
ModeManager, ensuring full isolation between tests.
"""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

from finalayze.api.v1.system import ModeManager, get_mode_manager
from finalayze.main import create_app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENV_VAR_REAL_CONFIRMED = "FINALAYZE_REAL_CONFIRMED"
API_VERSION = "0.1.0"

STATUS_OK = "ok"

MODE_DEBUG = "debug"
MODE_SANDBOX = "sandbox"
MODE_REAL = "real"
MODE_INVALID = "invalid_mode"

HTTP_200 = 200
HTTP_400 = 400
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


def make_client(application: object) -> AsyncClient:
    """Create an async test client for the given ASGI app."""
    return AsyncClient(transport=ASGITransport(app=application), base_url="http://test")  # type: ignore[arg-type]


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
        async with make_client(app) as client:
            response = await client.get("/api/v1/health")
        assert response.json()["status"] == STATUS_OK

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
        async with make_client(app) as client:
            response = await client.get("/api/v1/mode")
        assert response.status_code == HTTP_200

    @pytest.mark.asyncio
    async def test_get_mode_returns_debug_by_default(self) -> None:
        app, _ = build_test_app()
        async with make_client(app) as client:
            response = await client.get("/api/v1/mode")
        assert response.json()["mode"] == MODE_DEBUG


# ---------------------------------------------------------------------------
# POST /api/v1/mode
# ---------------------------------------------------------------------------
class TestSetModeEndpoint:
    @pytest.mark.asyncio
    async def test_set_mode_to_sandbox_returns_200(self) -> None:
        app, _ = build_test_app()
        async with make_client(app) as client:
            response = await client.post("/api/v1/mode", json={"mode": MODE_SANDBOX})
        assert response.status_code == HTTP_200

    @pytest.mark.asyncio
    async def test_set_mode_to_sandbox_changes_mode(self) -> None:
        app, _ = build_test_app()
        async with make_client(app) as client:
            await client.post("/api/v1/mode", json={"mode": MODE_SANDBOX})
            response = await client.get("/api/v1/mode")
        assert response.json()["mode"] == MODE_SANDBOX

    @pytest.mark.asyncio
    async def test_set_mode_response_contains_new_mode(self) -> None:
        app, _ = build_test_app()
        async with make_client(app) as client:
            response = await client.post("/api/v1/mode", json={"mode": MODE_SANDBOX})
        assert response.json()["mode"] == MODE_SANDBOX

    @pytest.mark.asyncio
    async def test_set_mode_to_real_without_env_returns_400(self) -> None:
        os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)
        app, _ = build_test_app()
        async with make_client(app) as client:
            response = await client.post("/api/v1/mode", json={"mode": MODE_REAL})
        assert response.status_code == HTTP_400

    @pytest.mark.asyncio
    async def test_set_mode_to_real_with_env_returns_200(self) -> None:
        os.environ[ENV_VAR_REAL_CONFIRMED] = "true"
        try:
            app, _ = build_test_app()
            async with make_client(app) as client:
                response = await client.post("/api/v1/mode", json={"mode": MODE_REAL})
            assert response.status_code == HTTP_200
        finally:
            os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)

    @pytest.mark.asyncio
    async def test_set_mode_to_real_error_detail_present(self) -> None:
        os.environ.pop(ENV_VAR_REAL_CONFIRMED, None)
        app, _ = build_test_app()
        async with make_client(app) as client:
            response = await client.post("/api/v1/mode", json={"mode": MODE_REAL})
        assert "detail" in response.json()

    @pytest.mark.asyncio
    async def test_set_invalid_mode_returns_422(self) -> None:
        app, _ = build_test_app()
        async with make_client(app) as client:
            response = await client.post("/api/v1/mode", json={"mode": MODE_INVALID})
        assert response.status_code == HTTP_422
