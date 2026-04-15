"""Tests for SentimentStore (data/sentiment_store.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from finalayze.data.sentiment_store import SentimentRow, SentimentStore


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session usable as async context manager."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture
def mock_factory(mock_session: AsyncMock) -> MagicMock:
    """Create a mock session factory that returns mock_session."""
    factory = MagicMock()
    factory.return_value = mock_session
    return factory


@pytest.fixture
def store(mock_factory: MagicMock) -> SentimentStore:
    """Create a SentimentStore with mocked session factory."""
    return SentimentStore(mock_factory)


def _make_row(
    bucket: datetime,
    avg_score: float | None,
    article_count: int,
) -> MagicMock:
    """Create a mock DB row with named attributes."""
    row = MagicMock()
    row.bucket = bucket
    row.avg_score = avg_score
    row.article_count = article_count
    return row


class TestSentimentStoreGetRolling:
    """Tests for SentimentStore.get_rolling()."""

    async def test_get_rolling_returns_sentiment_rows(
        self,
        store: SentimentStore,
        mock_session: AsyncMock,
    ) -> None:
        """get_rolling returns a list of SentimentRow with correct field values."""
        dt1 = datetime(2026, 4, 10, 0, 0, tzinfo=UTC)
        dt2 = datetime(2026, 4, 11, 0, 0, tzinfo=UTC)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            _make_row(dt1, 0.65, 12),
            _make_row(dt2, 0.72, 8),
        ]
        mock_session.execute.return_value = mock_result

        result = await store.get_rolling("SBER", window="7d")

        assert len(result) == 2
        assert isinstance(result[0], SentimentRow)
        assert result[0].bucket == dt1
        assert result[0].avg_score == pytest.approx(0.65)
        assert result[0].article_count == 12
        assert result[1].bucket == dt2
        assert result[1].avg_score == pytest.approx(0.72)
        assert result[1].article_count == 8

    async def test_get_rolling_empty_ticker_returns_empty_list(
        self,
        store: SentimentStore,
        mock_session: AsyncMock,
    ) -> None:
        """get_rolling on a ticker with no data returns an empty list without error."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        result = await store.get_rolling("UNKNOWN")

        assert result == []

    async def test_get_rolling_null_avg_score_handled(
        self,
        store: SentimentStore,
        mock_session: AsyncMock,
    ) -> None:
        """get_rolling handles NULL avg_score by returning None, not casting to float."""
        dt1 = datetime(2026, 4, 10, 0, 0, tzinfo=UTC)
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [_make_row(dt1, None, 3)]
        mock_session.execute.return_value = mock_result

        result = await store.get_rolling("SBER")

        assert len(result) == 1
        assert result[0].avg_score is None

    async def test_get_rolling_default_window_is_7d(
        self,
        store: SentimentStore,
        mock_session: AsyncMock,
    ) -> None:
        """Calling get_rolling without window kwarg uses '7 days' interval."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        await store.get_rolling("SBER")

        call_args = mock_session.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        assert params["interval"] == "7 days"

    async def test_get_rolling_30d_window(
        self,
        store: SentimentStore,
        mock_session: AsyncMock,
    ) -> None:
        """get_rolling with window='30d' passes '30 days' interval."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        await store.get_rolling("SBER", window="30d")

        call_args = mock_session.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        assert params["interval"] == "30 days"

    async def test_get_rolling_invalid_window_falls_back(
        self,
        store: SentimentStore,
        mock_session: AsyncMock,
    ) -> None:
        """get_rolling with invalid window falls back to '7 days' interval."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        await store.get_rolling("SBER", window="99d")

        call_args = mock_session.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        assert params["interval"] == "7 days"
