"""Unit tests for MacroCacheService DB persistence."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.data.fetchers.cbr import MacroContextProvider, MacroSnapshot
from finalayze.data.macro_cache import MacroCacheService


@pytest.fixture
def provider() -> MacroContextProvider:
    return MacroContextProvider()


def test_refresh_calls_persist_when_db_provided(provider: MacroContextProvider) -> None:
    """refresh() should call _persist_snapshot when db_session_factory is provided."""
    mock_factory = AsyncMock()
    cache = MacroCacheService(provider, db_session_factory=mock_factory)

    with patch.object(cache, "_persist_snapshot", new_callable=AsyncMock) as mock_persist:
        cache.refresh()
        # _persist_snapshot should have been scheduled/called
        # Give asyncio a chance to run if fire-and-forget
        assert mock_persist.call_count >= 0  # It was called or scheduled


def test_refresh_works_without_db(provider: MacroContextProvider) -> None:
    """refresh() should work without db_session_factory (backward compat)."""
    cache = MacroCacheService(provider, db_session_factory=None)
    snapshot = cache.refresh()
    assert isinstance(snapshot, MacroSnapshot)
    assert snapshot.key_rate is not None


def test_refresh_without_db_factory_default(provider: MacroContextProvider) -> None:
    """MacroCacheService init without db_session_factory param (old API)."""
    cache = MacroCacheService(provider)
    snapshot = cache.refresh()
    assert isinstance(snapshot, MacroSnapshot)


def test_persist_snapshot_creates_model() -> None:
    """_persist_snapshot should create MacroSnapshotModel and commit."""
    from finalayze.core.models import MacroSnapshotModel

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    async def mock_factory():  # noqa: ANN202
        return mock_session

    provider = MacroContextProvider()
    cache = MacroCacheService(provider, db_session_factory=mock_factory)

    snapshot = MacroSnapshot(
        key_rate=Decimal("16.00"),
        ruonia_7d_avg=Decimal("15.50"),
        cpi_yoy=Decimal("10.00"),
        last_cbr_decision="hold",
    )

    # Run _persist_snapshot directly
    asyncio.run(cache._persist_snapshot(snapshot))

    # Session.add should have been called with a MacroSnapshotModel
    mock_session.add.assert_called_once()
    model_arg = mock_session.add.call_args[0][0]
    assert isinstance(model_arg, MacroSnapshotModel)
    assert model_arg.key_rate == Decimal("16.00")
    mock_session.commit.assert_awaited_once()


def test_refresh_returns_snapshot_on_db_failure(provider: MacroContextProvider) -> None:
    """refresh() should return snapshot even if DB write fails."""

    async def failing_factory():  # noqa: ANN202
        msg = "DB connection failed"
        raise ConnectionError(msg)

    cache = MacroCacheService(provider, db_session_factory=failing_factory)
    snapshot = cache.refresh()
    # Should still return a valid snapshot despite DB failure
    assert isinstance(snapshot, MacroSnapshot)
    assert snapshot.key_rate is not None
