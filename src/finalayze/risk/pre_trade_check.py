"""Pre-trade risk checks (Layer 4).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.risk.circuit_breaker import CircuitLevel

# US market hours in UTC: 9:30 ET = 14:30 UTC, 16:00 ET = 21:00 UTC
_US_MARKET_OPEN_UTC_HOUR = 14
_US_MARKET_OPEN_UTC_MINUTE = 30
_US_MARKET_CLOSE_UTC_HOUR = 21
_US_MARKET_CLOSE_UTC_MINUTE = 0

# MOEX market hours in UTC: 10:00 MSK = 07:00 UTC, 18:45 MSK = 15:45 UTC
_MOEX_MARKET_OPEN_UTC_HOUR = 7
_MOEX_MARKET_OPEN_UTC_MINUTE = 0
_MOEX_MARKET_CLOSE_UTC_HOUR = 15
_MOEX_MARKET_CLOSE_UTC_MINUTE = 45

# Halted circuit breaker levels
_HALTING_LEVELS = frozenset({"halted", "liquidate"})

# Weekend weekday threshold (Saturday=5, Sunday=6)
_WEEKEND_WEEKDAY = 5

# PDT rule constants
_PDT_MAX_DAY_TRADES = 3
_PDT_ROLLING_DAYS = 5
_PDT_EQUITY_THRESHOLD = Decimal(25000)


@dataclass(frozen=True)
class PreTradeResult:
    """Result of pre-trade risk validation."""

    passed: bool
    violations: list[str] = field(default_factory=list)


class PDTTracker:
    """Track day-trade count over a 5-business-day rolling window.

    FINRA Pattern Day Trader rule: accounts with equity < $25,000 are
    limited to 3 day trades per 5 rolling business days.

    A "day trade" is defined as opening and closing the same position
    on the same trading day.
    """

    def __init__(self) -> None:
        # Store dates of day trades (most recent last)
        self._day_trade_dates: deque[date] = deque()

    def record_day_trade(self, trade_date: date) -> None:
        """Record that a day trade occurred on the given date."""
        self._day_trade_dates.append(trade_date)

    def _count_recent_day_trades(self, as_of: date) -> int:
        """Count day trades in the 5-business-day window ending on *as_of*.

        Business days are approximated by counting calendar days back to
        cover 5 weekdays (typically 7 calendar days).  We use a conservative
        7-calendar-day window to avoid missing trades near weekends.
        """
        cutoff = as_of - timedelta(days=7)
        # Purge stale entries older than the window
        while self._day_trade_dates and self._day_trade_dates[0] < cutoff:
            self._day_trade_dates.popleft()
        return sum(1 for d in self._day_trade_dates if d >= cutoff)

    def would_violate(self, as_of: date, account_equity: Decimal) -> bool:
        """Return True if executing another day trade would violate PDT.

        Accounts with equity >= $25,000 are exempt from the PDT rule.
        Non-US markets are never subject to PDT.
        """
        if account_equity >= _PDT_EQUITY_THRESHOLD:
            return False
        recent = self._count_recent_day_trades(as_of)
        return recent >= _PDT_MAX_DAY_TRADES

    @property
    def recent_day_trades(self) -> int:
        """Number of day trades currently tracked (informational)."""
        return len(self._day_trade_dates)


class PreTradeChecker:
    """Validates orders against risk limits before execution.

    Implements all 11 required pre-trade checks:
        1. Market hours check (per market)
        2. Symbol validity (symbol exists in market) — caller responsibility
        3. Mode allows order — caller responsibility
        4. Circuit breaker status (per market)
        5. PDT compliance (US only, accounts < $25K)
        6. Position size (Kelly + max cap)
        7. Portfolio rules (max positions, sector concentration)
        8. Cash sufficient (per market/currency)
        9. Stop-loss must be set (when require_stop_loss=True)
        10. No duplicate pending order
        11. Cross-market exposure limit
    """

    def __init__(
        self,
        max_position_pct: Decimal = Decimal("0.20"),
        max_positions_per_market: int = 10,
        pdt_tracker: PDTTracker | None = None,
        max_sector_concentration_pct: Decimal = Decimal("0.40"),
        min_cash_reserve_pct: Decimal = Decimal("0.20"),
    ) -> None:
        self._max_position_pct = max_position_pct
        self._max_positions = max_positions_per_market
        self._pdt_tracker = pdt_tracker
        self._max_sector_pct = max_sector_concentration_pct
        self._min_cash_reserve_pct = min_cash_reserve_pct

    def check(  # noqa: PLR0912
        self,
        order_value: Decimal,
        portfolio_equity: Decimal,
        available_cash: Decimal,
        open_position_count: int,
        market_id: str = "us",
        dt: datetime | None = None,
        circuit_breaker_level: CircuitLevel | None = None,
        stop_loss_price: Decimal | None = None,
        require_stop_loss: bool = False,
        has_pending_order: bool = False,
        symbol: str = "",
        cross_market_exposure_pct: Decimal | None = None,
        max_cross_market_exposure_pct: Decimal | None = None,
        is_day_trade: bool = False,
        sector_exposure_value: Decimal | None = None,
        sector_id: str = "",
    ) -> PreTradeResult:
        """Run all pre-trade risk checks.

        Args:
            order_value: Notional value of the proposed order.
            portfolio_equity: Current total portfolio equity.
            available_cash: Cash available for trading.
            open_position_count: Number of currently open positions.
            market_id: Market identifier ("us" or "moex").
            dt: Current UTC datetime for market hours check. Uses now() if None.
            circuit_breaker_level: Current circuit breaker level for this market.
            stop_loss_price: Stop-loss price for the order (None if not set).
            require_stop_loss: Whether a stop-loss price is required.
            has_pending_order: Whether there is already a pending order for symbol.
            symbol: The symbol being ordered (for duplicate check).
            cross_market_exposure_pct: Current cross-market exposure fraction.
            max_cross_market_exposure_pct: Maximum allowed cross-market exposure.
            is_day_trade: Whether this order would constitute a day trade.

        Returns:
            A :class:`PreTradeResult` indicating pass/fail and any violations.
        """
        violations: list[str] = []

        # 1. Market hours check
        check_dt = dt if dt is not None else datetime.now(UTC)
        if not self._is_market_open(market_id, check_dt):
            violations.append(
                f"Market '{market_id}' is closed at {check_dt.strftime('%Y-%m-%d %H:%M UTC')}"
            )

        # 4. Circuit breaker status
        if circuit_breaker_level is not None:
            level_str = str(circuit_breaker_level).lower()
            if level_str in _HALTING_LEVELS:
                violations.append(
                    f"Circuit breaker is {level_str} for market '{market_id}' — trading halted"
                )

        # 5. PDT compliance (US only)
        if (
            market_id == "us"
            and is_day_trade
            and self._pdt_tracker is not None
            and self._pdt_tracker.would_violate(check_dt.date(), portfolio_equity)
        ):
            recent = self._pdt_tracker.recent_day_trades
            violations.append(
                f"PDT violation: {recent} day trades in last 5 business days "
                f"(max {_PDT_MAX_DAY_TRADES}), equity ${float(portfolio_equity):,.0f} "
                f"< ${float(_PDT_EQUITY_THRESHOLD):,.0f}"
            )

        # 6. Position size check
        if portfolio_equity == 0:
            violations.append("Portfolio equity is zero; no trades permitted")
        else:
            pct = order_value / portfolio_equity
            if pct > self._max_position_pct:
                max_pct = float(self._max_position_pct)
                violations.append(f"Position size {float(pct):.1%} exceeds max {max_pct:.1%}")

        # 7. Portfolio rules — max positions
        if open_position_count >= self._max_positions:
            violations.append(
                f"Open positions ({open_position_count}) >= max ({self._max_positions})"
            )

        # 7b. Sector/segment concentration
        if sector_exposure_value is not None and portfolio_equity > 0 and sector_id:
            prospective = sector_exposure_value + order_value
            concentration = prospective / portfolio_equity
            if concentration > self._max_sector_pct:
                violations.append(
                    f"Sector '{sector_id}' concentration {float(concentration):.1%} "
                    f"exceeds max {float(self._max_sector_pct):.1%}"
                )

        # 8. Cash sufficient
        if order_value > available_cash:
            violations.append(f"Insufficient cash: need {order_value}, have {available_cash}")

        # 8b. Cash reserve check
        if portfolio_equity > 0:
            post_trade_cash = available_cash - order_value
            reserve_ratio = post_trade_cash / portfolio_equity
            if reserve_ratio < self._min_cash_reserve_pct:
                violations.append(
                    f"Post-trade cash reserve {float(reserve_ratio):.1%} "
                    f"below min {float(self._min_cash_reserve_pct):.1%}"
                )

        # 9. Stop-loss must be set
        if require_stop_loss and stop_loss_price is None:
            violations.append("Stop-loss price is required but not set")

        # 10. No duplicate pending order
        if has_pending_order and symbol:
            violations.append(f"Duplicate pending order for {symbol}")

        # 11. Cross-market exposure limit
        if (
            cross_market_exposure_pct is not None
            and max_cross_market_exposure_pct is not None
            and cross_market_exposure_pct > max_cross_market_exposure_pct
        ):
            violations.append(
                f"Cross-market exposure {float(cross_market_exposure_pct):.1%} "
                f"exceeds max {float(max_cross_market_exposure_pct):.1%}"
            )

        return PreTradeResult(passed=len(violations) == 0, violations=violations)

    @staticmethod
    def _is_market_open(market_id: str, dt: datetime) -> bool:
        """Return True if the market is open at the given UTC datetime."""
        # Weekends: Saturday=5, Sunday=6
        if dt.weekday() >= _WEEKEND_WEEKDAY:
            return False

        if market_id == "us":
            open_minutes = _US_MARKET_OPEN_UTC_HOUR * 60 + _US_MARKET_OPEN_UTC_MINUTE
            close_minutes = _US_MARKET_CLOSE_UTC_HOUR * 60 + _US_MARKET_CLOSE_UTC_MINUTE
        elif market_id == "moex":
            open_minutes = _MOEX_MARKET_OPEN_UTC_HOUR * 60 + _MOEX_MARKET_OPEN_UTC_MINUTE
            close_minutes = _MOEX_MARKET_CLOSE_UTC_HOUR * 60 + _MOEX_MARKET_CLOSE_UTC_MINUTE
        else:
            # Unknown market: assume open
            return True

        current_minutes = dt.hour * 60 + dt.minute
        return open_minutes <= current_minutes < close_minutes
