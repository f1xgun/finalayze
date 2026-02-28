"""Tests for health check logic in api/v1/system.py."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.api.v1.system import (
    ComponentStatus,
    _check_db,
    _check_redis,
    _get_component_status,
)


class TestCheckDb:
    """Tests for _check_db probe."""

    @pytest.mark.asyncio
    async def test_db_healthy(self) -> None:
        mock_conn = AsyncMock()

        class _FakeCtx:
            async def __aenter__(self) -> AsyncMock:
                return mock_conn

            async def __aexit__(self, *args: object) -> None:
                pass

        mock_engine = MagicMock()
        mock_engine.connect.return_value = _FakeCtx()
        mock_engine.dispose = AsyncMock()

        with patch("finalayze.api.v1.system.create_async_engine", return_value=mock_engine):
            result = await _check_db()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_db_down(self) -> None:
        with patch(
            "finalayze.api.v1.system.create_async_engine",
            side_effect=Exception("connection refused"),
        ):
            result = await _check_db()
        assert result == "error"


class TestCheckRedis:
    """Tests for _check_redis probe."""

    @pytest.mark.asyncio
    async def test_redis_healthy(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()

        with patch("finalayze.api.v1.system.redis.asyncio.from_url", return_value=mock_redis):
            result = await _check_redis()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_redis_timeout(self) -> None:
        with patch(
            "finalayze.api.v1.system.redis.asyncio.from_url",
            side_effect=TimeoutError("timeout"),
        ):
            result = await _check_redis()
        assert result == "error"


class TestGetComponentStatus:
    """Tests for cached _get_component_status."""

    @pytest.mark.asyncio
    async def test_all_healthy(self) -> None:
        import finalayze.api.v1.system as sys_mod

        sys_mod._health_cache = {}
        sys_mod._health_cache_ts = 0.0

        with (
            patch.object(sys_mod, "_check_db", new_callable=AsyncMock, return_value="ok"),
            patch.object(sys_mod, "_check_redis", new_callable=AsyncMock, return_value="ok"),
        ):
            status = await _get_component_status()

        assert status.db == "ok"
        assert status.redis == "ok"

    @pytest.mark.asyncio
    async def test_db_down_returns_error(self) -> None:
        import finalayze.api.v1.system as sys_mod

        sys_mod._health_cache = {}
        sys_mod._health_cache_ts = 0.0

        with (
            patch.object(sys_mod, "_check_db", new_callable=AsyncMock, return_value="error"),
            patch.object(sys_mod, "_check_redis", new_callable=AsyncMock, return_value="ok"),
        ):
            status = await _get_component_status()

        assert status.db == "error"
        assert status.redis == "ok"

    @pytest.mark.asyncio
    async def test_cache_returns_stale_within_ttl(self) -> None:
        import finalayze.api.v1.system as sys_mod

        sys_mod._health_cache = {
            "db": "ok",
            "redis": "ok",
            "alpaca": "ok",
            "tinkoff": "ok",
            "llm": "ok",
        }
        sys_mod._health_cache_ts = time.monotonic()  # fresh cache

        check_db = AsyncMock()
        check_redis = AsyncMock()
        with (
            patch.object(sys_mod, "_check_db", check_db),
            patch.object(sys_mod, "_check_redis", check_redis),
        ):
            status = await _get_component_status()

        assert status.db == "ok"
        # Probes should NOT have been called (cache hit)
        check_db.assert_not_called()
        check_redis.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_expired_refreshes(self) -> None:
        import finalayze.api.v1.system as sys_mod

        sys_mod._health_cache = {
            "db": "ok",
            "redis": "ok",
            "alpaca": "ok",
            "tinkoff": "ok",
            "llm": "ok",
        }
        # Simulate expired cache
        sys_mod._health_cache_ts = time.monotonic() - 60

        with (
            patch.object(sys_mod, "_check_db", new_callable=AsyncMock, return_value="error"),
            patch.object(sys_mod, "_check_redis", new_callable=AsyncMock, return_value="ok"),
        ):
            status = await _get_component_status()

        assert status.db == "error"
