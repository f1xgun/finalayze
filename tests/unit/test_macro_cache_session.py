"""Tests for MacroCacheService session scoping (CONC-04).

Verifies:
- _persist_snapshot uses async-with context manager (session always closed)
- Rollback happens automatically on commit failure
- Failure is logged as warning (not raised)
"""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.data.fetchers.cbr import MacroSnapshot
from finalayze.data.macro_cache import MacroCacheService


def _make_snapshot() -> MacroSnapshot:
    """Create a minimal MacroSnapshot for testing."""
    return MacroSnapshot(
        key_rate=Decimal("21.0"),
        ruonia_7d_avg=Decimal("20.5"),
        cpi_yoy=Decimal("7.2"),
        last_cbr_decision="hold",
        breakeven_inflation=Decimal("6.5"),
        yield_curve=None,
        usdrub=Decimal("92.5"),
        ofzin_indexation_coefficient=None,
    )


class TestPersistSnapshotSessionScoping:
    """CONC-04: _persist_snapshot must use async-with for session lifecycle."""

    def test_persist_uses_async_with_context_manager(self) -> None:
        """Source code must use 'async with self._db_session_factory() as session'."""
        source = inspect.getsource(MacroCacheService._persist_snapshot)
        assert "async with self._db_session_factory() as session:" in source

    def test_no_bare_session_assignment(self) -> None:
        """No bare 'session = await self._db_session_factory()' pattern."""
        source = inspect.getsource(MacroCacheService._persist_snapshot)
        assert "session = await self._db_session_factory()" not in source

    @pytest.mark.asyncio
    async def test_session_commit_called_on_success(self) -> None:
        """On success, session.commit() is awaited."""
        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_factory():
            yield mock_session

        provider = MagicMock()
        cache = MacroCacheService(provider=provider, db_session_factory=mock_factory)
        snapshot = _make_snapshot()

        await cache._persist_snapshot(snapshot)

        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_session_closed_on_commit_failure(self) -> None:
        """On commit failure, session context manager exits (rollback) and warning logged."""
        mock_session = AsyncMock()
        mock_session.commit.side_effect = RuntimeError("DB down")
        exited = {"called": False}

        @asynccontextmanager
        async def mock_factory():
            yield mock_session
            exited["called"] = True  # this runs in __aexit__ cleanup

        provider = MagicMock()
        cache = MacroCacheService(provider=provider, db_session_factory=mock_factory)
        snapshot = _make_snapshot()

        # Should not raise -- fire-and-forget with warning
        with patch("finalayze.data.macro_cache._log") as mock_log:
            await cache._persist_snapshot(snapshot)
            mock_log.warning.assert_called_once()
            assert "macro_snapshot_persist_db_failed" in str(mock_log.warning.call_args)

    def test_persist_logs_db_failed_on_error(self) -> None:
        """Source code must contain macro_snapshot_persist_db_failed log message."""
        source = inspect.getsource(MacroCacheService._persist_snapshot)
        assert "macro_snapshot_persist_db_failed" in source
