"""Unit tests for PDT (Pattern Day Trader) tracking and pre-trade check integration."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from finalayze.risk.pre_trade_check import (
    _PDT_EQUITY_THRESHOLD,
    _PDT_MAX_DAY_TRADES,
    PDTTracker,
    PreTradeChecker,
)

# ── Constants (ruff PLR2004) ────────────────────────────────────────────
_SMALL_ACCOUNT = Decimal(20000)
_LARGE_ACCOUNT = Decimal(30000)
_TRADE_VALUE = Decimal(2000)
_CASH = Decimal(10000)
_POSITIONS = 1
_MONDAY = date(2026, 2, 23)  # a Monday


class TestPDTTrackerBasics:
    """Basic PDT tracker functionality."""

    def test_empty_tracker_allows_trades(self) -> None:
        tracker = PDTTracker()
        assert not tracker.would_violate(_MONDAY, _SMALL_ACCOUNT)

    def test_recent_day_trades_starts_at_zero(self) -> None:
        tracker = PDTTracker()
        assert tracker.recent_day_trades == 0

    def test_record_increments_count(self) -> None:
        tracker = PDTTracker()
        tracker.record_day_trade(_MONDAY)
        assert tracker.recent_day_trades == 1


class TestPDTRuleEnforcement:
    """PDT rule: max 3 day trades per 5 rolling business days for accounts < $25K."""

    def test_three_trades_still_allowed(self) -> None:
        tracker = PDTTracker()
        for i in range(_PDT_MAX_DAY_TRADES):
            tracker.record_day_trade(_MONDAY + timedelta(days=i))
        # 3 trades recorded, 4th would violate
        assert tracker.would_violate(_MONDAY + timedelta(days=3), _SMALL_ACCOUNT)

    def test_two_trades_allowed(self) -> None:
        tracker = PDTTracker()
        two_trades = 2
        for i in range(two_trades):
            tracker.record_day_trade(_MONDAY + timedelta(days=i))
        assert not tracker.would_violate(_MONDAY + timedelta(days=2), _SMALL_ACCOUNT)

    def test_large_account_exempt(self) -> None:
        """Accounts >= $25K are exempt from PDT."""
        tracker = PDTTracker()
        for i in range(_PDT_MAX_DAY_TRADES):
            tracker.record_day_trade(_MONDAY + timedelta(days=i))
        assert not tracker.would_violate(_MONDAY + timedelta(days=3), _LARGE_ACCOUNT)

    def test_exactly_at_threshold_exempt(self) -> None:
        tracker = PDTTracker()
        for i in range(_PDT_MAX_DAY_TRADES):
            tracker.record_day_trade(_MONDAY + timedelta(days=i))
        assert not tracker.would_violate(_MONDAY + timedelta(days=3), _PDT_EQUITY_THRESHOLD)

    def test_old_trades_expire(self) -> None:
        """Day trades older than the rolling window don't count."""
        tracker = PDTTracker()
        old_date = _MONDAY - timedelta(days=10)
        for i in range(_PDT_MAX_DAY_TRADES):
            tracker.record_day_trade(old_date + timedelta(days=i))
        # All trades are > 7 days old — should not block
        assert not tracker.would_violate(_MONDAY, _SMALL_ACCOUNT)


class TestPDTInPreTradeChecker:
    """PDT check integrated into PreTradeChecker."""

    def _make_checker(self, tracker: PDTTracker) -> PreTradeChecker:
        return PreTradeChecker(pdt_tracker=tracker)

    def test_pdt_violation_blocks_trade(self) -> None:
        tracker = PDTTracker()
        for i in range(_PDT_MAX_DAY_TRADES):
            tracker.record_day_trade(_MONDAY + timedelta(days=i))

        checker = self._make_checker(tracker)
        from datetime import UTC, datetime

        dt = datetime(_MONDAY.year, _MONDAY.month, _MONDAY.day + 3, 15, 0, tzinfo=UTC)
        result = checker.check(
            order_value=_TRADE_VALUE,
            portfolio_equity=_SMALL_ACCOUNT,
            available_cash=_CASH,
            open_position_count=_POSITIONS,
            market_id="us",
            dt=dt,
            is_day_trade=True,
        )
        assert not result.passed
        assert any("PDT" in v for v in result.violations)

    def test_non_day_trade_skips_pdt(self) -> None:
        """Regular (non-day) trades don't trigger PDT check."""
        tracker = PDTTracker()
        for i in range(_PDT_MAX_DAY_TRADES):
            tracker.record_day_trade(_MONDAY + timedelta(days=i))

        checker = self._make_checker(tracker)
        from datetime import UTC, datetime

        dt = datetime(_MONDAY.year, _MONDAY.month, _MONDAY.day + 3, 15, 0, tzinfo=UTC)
        result = checker.check(
            order_value=_TRADE_VALUE,
            portfolio_equity=_SMALL_ACCOUNT,
            available_cash=_CASH,
            open_position_count=_POSITIONS,
            market_id="us",
            dt=dt,
            is_day_trade=False,
        )
        # Should pass (PDT only applies to day trades)
        assert result.passed

    def test_moex_skips_pdt(self) -> None:
        """MOEX market is not subject to US PDT rules."""
        tracker = PDTTracker()
        for i in range(_PDT_MAX_DAY_TRADES):
            tracker.record_day_trade(_MONDAY + timedelta(days=i))

        checker = self._make_checker(tracker)
        from datetime import UTC, datetime

        dt = datetime(_MONDAY.year, _MONDAY.month, _MONDAY.day + 3, 8, 0, tzinfo=UTC)
        result = checker.check(
            order_value=_TRADE_VALUE,
            portfolio_equity=_SMALL_ACCOUNT,
            available_cash=_CASH,
            open_position_count=_POSITIONS,
            market_id="moex",
            dt=dt,
            is_day_trade=True,
        )
        assert result.passed

    def test_no_tracker_skips_pdt(self) -> None:
        """When no PDTTracker is provided, check 5 is silently skipped."""
        checker = PreTradeChecker()
        from datetime import UTC, datetime

        dt = datetime(_MONDAY.year, _MONDAY.month, _MONDAY.day, 15, 0, tzinfo=UTC)
        result = checker.check(
            order_value=_TRADE_VALUE,
            portfolio_equity=_SMALL_ACCOUNT,
            available_cash=_CASH,
            open_position_count=_POSITIONS,
            market_id="us",
            dt=dt,
            is_day_trade=True,
        )
        assert result.passed
