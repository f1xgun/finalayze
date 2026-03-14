"""Tests for TradingLoop._is_market_open holiday gate (01-01)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.trading_loop import TradingLoop


@pytest.fixture()
def mock_loop() -> MagicMock:
    """Create a MagicMock that can call _is_market_open as unbound."""
    return MagicMock(spec=[])


class TestMoexHolidayGate:
    """_is_market_open should return False for MOEX holidays during market hours."""

    def test_transferred_holiday_returns_false(self, mock_loop: MagicMock) -> None:
        """2024-04-29 is a transferred holiday -- MOEX closed even during market hours."""
        dt = datetime(2024, 4, 29, 10, 0, tzinfo=UTC)  # Monday, 10:00 UTC
        result = TradingLoop._is_market_open(mock_loop, "moex", dt)
        assert result is False

    def test_fixed_holiday_returns_false(self, mock_loop: MagicMock) -> None:
        """2024-01-01 is New Year's Day -- MOEX closed."""
        dt = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        result = TradingLoop._is_market_open(mock_loop, "moex", dt)
        assert result is False

    def test_normal_monday_returns_true(self, mock_loop: MagicMock) -> None:
        """2024-03-11 is a normal Monday -- MOEX open during market hours."""
        dt = datetime(2024, 3, 11, 10, 0, tzinfo=UTC)  # Within MOEX hours
        result = TradingLoop._is_market_open(mock_loop, "moex", dt)
        assert result is True

    def test_weekend_still_returns_false(self, mock_loop: MagicMock) -> None:
        """Weekend check still works for MOEX."""
        dt = datetime(2024, 3, 16, 10, 0, tzinfo=UTC)  # Saturday
        result = TradingLoop._is_market_open(mock_loop, "moex", dt)
        assert result is False


class TestUsMarketUnchanged:
    """US market behavior should NOT be affected by MOEX holiday changes."""

    def test_us_weekday_during_hours(self, mock_loop: MagicMock) -> None:
        """US market open on a normal weekday during market hours."""
        dt = datetime(2024, 3, 11, 15, 0, tzinfo=UTC)  # Monday 15:00 UTC (within US hours)
        result = TradingLoop._is_market_open(mock_loop, "us", dt)
        assert result is True

    def test_us_weekend_returns_false(self, mock_loop: MagicMock) -> None:
        """US weekend check still works."""
        dt = datetime(2024, 3, 16, 15, 0, tzinfo=UTC)  # Saturday
        result = TradingLoop._is_market_open(mock_loop, "us", dt)
        assert result is False

    def test_us_no_moex_holiday_check(self, mock_loop: MagicMock) -> None:
        """US market should NOT check MOEX holidays -- Jan 1 during US hours is not gated."""
        # 2024-01-01 is a Monday and MOEX holiday, but US holiday check is not wired
        dt = datetime(2024, 1, 1, 15, 0, tzinfo=UTC)
        result = TradingLoop._is_market_open(mock_loop, "us", dt)
        # US doesn't have holiday check in TradingLoop -- it returns True during hours
        assert result is True


class TestBondCycleHolidayGating:
    """Bond cycle gating: skip on MOEX holidays, skip outside hours, proceed when OK."""

    def test_bond_cycle_skips_moex_holiday(self) -> None:
        """_bond_cycle skips when is_moex_trading_day returns False (structlog, no Telegram)."""
        loop = MagicMock()
        loop._bond_processor = MagicMock()
        loop._bond_enabled = True
        loop._now.return_value = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)

        with patch(
            "finalayze.core.trading_loop.is_moex_trading_day", return_value=False
        ):
            TradingLoop._bond_cycle(loop)

        loop._bond_processor.run_cycle.assert_not_called()
        loop._alerter.on_error.assert_not_called()

    def test_bond_cycle_skips_outside_market_hours(self) -> None:
        """_bond_cycle skips when outside 10:00-18:45 MSK (07:00-15:45 UTC)."""
        loop = MagicMock()
        loop._bond_processor = MagicMock()
        loop._bond_enabled = True
        # 2024-03-11 is a normal Monday, but 05:00 UTC is before MOEX open
        loop._now.return_value = datetime(2024, 3, 11, 5, 0, tzinfo=UTC)

        with patch(
            "finalayze.core.trading_loop.is_moex_trading_day", return_value=True
        ):
            TradingLoop._bond_cycle(loop)

        loop._bond_processor.run_cycle.assert_not_called()

    def test_bond_cycle_proceeds_when_ok(self) -> None:
        """_bond_cycle proceeds when MOEX trading day AND within market hours."""
        loop = MagicMock()
        loop._bond_processor = MagicMock()
        loop._bond_enabled = True
        # 2024-03-11 is normal Monday, 10:00 UTC is within MOEX hours
        loop._now.return_value = datetime(2024, 3, 11, 10, 0, tzinfo=UTC)
        loop._is_market_open.return_value = True

        with patch(
            "finalayze.core.trading_loop.is_moex_trading_day", return_value=True
        ):
            TradingLoop._bond_cycle(loop)

        loop._bond_processor.run_cycle.assert_called_once()

    def test_macro_refresh_runs_regardless_of_holiday(self) -> None:
        """_macro_refresh runs regardless of holiday (no gate)."""
        loop = MagicMock()
        loop._macro_cache = MagicMock()
        loop._macro_cache.refresh.return_value = MagicMock(key_rate="21.00", ruonia_7d_avg="20.50")

        TradingLoop._macro_refresh(loop)

        loop._macro_cache.refresh.assert_called_once()
