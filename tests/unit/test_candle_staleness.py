"""Unit tests for candle staleness detection in TradingLoop."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from finalayze.core.trading_loop import TradingLoop


class TestIsCandleStale:
    """Tests for TradingLoop._is_candle_stale() static method."""

    def test_fresh_candle_not_stale(self) -> None:
        """A candle from 30 minutes ago should not be stale (2h threshold)."""
        now = datetime.now(UTC)
        latest_ts = now - timedelta(minutes=30)
        threshold_hours = 2.0
        assert TradingLoop._is_candle_stale(latest_ts, threshold_hours) is False

    def test_old_candle_is_stale(self) -> None:
        """A candle from 3 hours ago should be stale (2h threshold)."""
        now = datetime.now(UTC)
        latest_ts = now - timedelta(hours=3)
        threshold_hours = 2.0
        assert TradingLoop._is_candle_stale(latest_ts, threshold_hours) is True

    def test_boundary_exactly_at_threshold(self) -> None:
        """A candle exactly at the threshold boundary should be stale (>=)."""
        now = datetime.now(UTC)
        latest_ts = now - timedelta(hours=2)
        threshold_hours = 2.0
        # At exactly 2 hours, it should be considered stale
        assert TradingLoop._is_candle_stale(latest_ts, threshold_hours) is True

    def test_bond_threshold_24_hours(self) -> None:
        """Bonds use 24h threshold. 12-hour-old candle should not be stale."""
        now = datetime.now(UTC)
        latest_ts = now - timedelta(hours=12)
        threshold_hours = 24.0
        assert TradingLoop._is_candle_stale(latest_ts, threshold_hours) is False

    def test_bond_threshold_stale(self) -> None:
        """Bonds use 24h threshold. 25-hour-old candle should be stale."""
        now = datetime.now(UTC)
        latest_ts = now - timedelta(hours=25)
        threshold_hours = 24.0
        assert TradingLoop._is_candle_stale(latest_ts, threshold_hours) is True

    def test_naive_datetime_treated_as_utc(self) -> None:
        """Naive datetimes should still work (compared against UTC now)."""
        latest_ts = datetime.now(UTC) - timedelta(hours=5)
        threshold_hours = 2.0
        assert TradingLoop._is_candle_stale(latest_ts, threshold_hours) is True
