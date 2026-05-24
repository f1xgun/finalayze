"""Pre-trade risk checks (Layer 4).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.

Interface: PreTradeChecker.check(ctx: CheckContext) → PreTradeResult
Each built-in guard implements PreTradeCheck: check(ctx) → str | None.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

_log = structlog.get_logger()

if TYPE_CHECKING:
    from finalayze.risk.circuit_breaker import CircuitLevel
    from finalayze.risk.regime import RegimeState

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

# Check 13: strategies requiring fresh parameters
_PARAM_FRESHNESS_STRATEGIES = frozenset({"ou_mean_reversion", "pairs"})
_MAX_PARAM_AGE_BARS = 5

# Check 14: max correlated positions
_MAX_CORRELATED_POSITIONS = 3
_CORRELATION_THRESHOLD = 0.7

# Weekend weekday threshold (Saturday=5, Sunday=6)
_WEEKEND_WEEKDAY = 5

# PDT rule constants
_PDT_MAX_DAY_TRADES = 3
_PDT_ROLLING_DAYS = 5
_PDT_EQUITY_THRESHOLD = Decimal(25000)


# ── Typed context ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckContext:
    """Typed context carrying all data needed by pre-trade checks.

    Replaces the 22-param ``PreTradeChecker.check()`` signature.  Callers
    build one context object; the registry fans out to each check.
    """

    order_value: Decimal
    portfolio_equity: Decimal
    available_cash: Decimal
    open_position_count: int
    market_id: str = "us"
    dt: datetime | None = None
    circuit_breaker_level: CircuitLevel | None = None
    stop_loss_price: Decimal | None = None
    require_stop_loss: bool = False
    has_pending_order: bool = False
    symbol: str = ""
    cross_market_exposure_pct: Decimal | None = None
    max_cross_market_exposure_pct: Decimal | None = None
    is_day_trade: bool = False
    sector_exposure_value: Decimal | None = None
    sector_id: str = ""
    markets_active: list[str] | None = None
    regime_state: RegimeState | None = None
    strategy_name: str | None = None
    param_age_bars: int | None = None
    open_positions: list[str] | None = None
    correlations: dict[tuple[str, str], float] | None = None


# ── Protocol ───────────────────────────────────────────────────────────────


@runtime_checkable
class PreTradeCheck(Protocol):
    """Single-method protocol for one pre-trade guard.

    Return the violation string on failure, ``None`` on pass.
    """

    def check(self, ctx: CheckContext) -> str | None: ...


# ── Result ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PreTradeResult:
    """Result of pre-trade risk validation."""

    passed: bool
    violations: list[str] = field(default_factory=list)


# ── PDT tracker (stateful, not a check) ───────────────────────────────────


class PDTTracker:
    """Track day-trade count over a 5-business-day rolling window.

    FINRA Pattern Day Trader rule: accounts with equity < $25,000 are
    limited to 3 day trades per 5 rolling business days.

    A "day trade" is defined as opening and closing the same position
    on the same trading day.
    """

    def __init__(self) -> None:
        self._day_trade_dates: deque[date] = deque()

    def record_day_trade(self, trade_date: date) -> None:
        """Record that a day trade occurred on the given date."""
        self._day_trade_dates.append(trade_date)

    def _count_recent_day_trades(self, as_of: date) -> int:
        cutoff = as_of - timedelta(days=7)
        while self._day_trade_dates and self._day_trade_dates[0] < cutoff:
            self._day_trade_dates.popleft()
        return sum(1 for d in self._day_trade_dates if d >= cutoff)

    def would_violate(self, as_of: date, account_equity: Decimal) -> bool:
        """Return True if executing another day trade would violate PDT."""
        if account_equity >= _PDT_EQUITY_THRESHOLD:
            return False
        recent = self._count_recent_day_trades(as_of)
        return recent >= _PDT_MAX_DAY_TRADES

    @property
    def recent_day_trades(self) -> int:
        """Number of day trades currently tracked (informational)."""
        return len(self._day_trade_dates)


# ── Shared helper ─────────────────────────────────────────────────────────


def _is_market_open(market_id: str, dt: datetime) -> bool:
    if dt.weekday() >= _WEEKEND_WEEKDAY:
        return False
    if market_id == "us":
        open_min = _US_MARKET_OPEN_UTC_HOUR * 60 + _US_MARKET_OPEN_UTC_MINUTE
        close_min = _US_MARKET_CLOSE_UTC_HOUR * 60 + _US_MARKET_CLOSE_UTC_MINUTE
    elif market_id == "moex":
        open_min = _MOEX_MARKET_OPEN_UTC_HOUR * 60 + _MOEX_MARKET_OPEN_UTC_MINUTE
        close_min = _MOEX_MARKET_CLOSE_UTC_HOUR * 60 + _MOEX_MARKET_CLOSE_UTC_MINUTE
    else:
        return True
    current = dt.hour * 60 + dt.minute
    return open_min <= current < close_min


# ── Built-in check classes ─────────────────────────────────────────────────


class MarketHoursCheck:
    """Check 1 — market must be open."""

    def check(self, ctx: CheckContext) -> str | None:
        dt = ctx.dt if ctx.dt is not None else datetime.now(UTC)
        if not _is_market_open(ctx.market_id, dt):
            return f"Market '{ctx.market_id}' is closed at {dt.strftime('%Y-%m-%d %H:%M UTC')}"
        return None


class CircuitBreakerCheck:
    """Check 4 — circuit breaker must not be in a halting state."""

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.circuit_breaker_level is None:
            return None
        level_str = str(ctx.circuit_breaker_level).lower()
        if level_str in _HALTING_LEVELS:
            return f"Circuit breaker is {level_str} for market '{ctx.market_id}' — trading halted"
        return None


class PDTCheck:
    """Check 5 — PDT compliance (US only, accounts < $25 K)."""

    def __init__(self, tracker: PDTTracker | None = None) -> None:
        self._tracker = tracker

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.market_id != "us" or not ctx.is_day_trade or self._tracker is None:
            return None
        dt = ctx.dt if ctx.dt is not None else datetime.now(UTC)
        if self._tracker.would_violate(dt.date(), ctx.portfolio_equity):
            recent = self._tracker.recent_day_trades
            return (
                f"PDT violation: {recent} day trades in last 5 business days "
                f"(max {_PDT_MAX_DAY_TRADES}), equity "
                f"${float(ctx.portfolio_equity):,.0f} "
                f"< ${float(_PDT_EQUITY_THRESHOLD):,.0f}"
            )
        return None


class PositionSizeCheck:
    """Check 6 — order must not exceed max position size."""

    def __init__(self, max_position_pct: Decimal = Decimal("0.20")) -> None:
        self._max_pct = max_position_pct

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.portfolio_equity == 0:
            return "Portfolio equity is zero; no trades permitted"
        pct = ctx.order_value / ctx.portfolio_equity
        if pct > self._max_pct:
            return f"Position size {float(pct):.1%} exceeds max {float(self._max_pct):.1%}"
        return None


class MaxPositionsCheck:
    """Check 7a — open positions must not reach the cap."""

    def __init__(self, max_positions: int = 10) -> None:
        self._max = max_positions

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.open_position_count >= self._max:
            return f"Open positions ({ctx.open_position_count}) >= max ({self._max})"
        return None


class SectorConcentrationCheck:
    """Check 7b — sector/segment concentration limit."""

    def __init__(self, max_sector_pct: Decimal = Decimal("0.40")) -> None:
        self._max = max_sector_pct

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.sector_exposure_value is None or not ctx.sector_id or ctx.portfolio_equity <= 0:
            return None
        concentration = (ctx.sector_exposure_value + ctx.order_value) / ctx.portfolio_equity
        if concentration > self._max:
            return (
                f"Sector '{ctx.sector_id}' concentration "
                f"{float(concentration):.1%} exceeds max "
                f"{float(self._max):.1%}"
            )
        return None


class CashCheck:
    """Check 8a — sufficient cash for the order."""

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.order_value > ctx.available_cash:
            return f"Insufficient cash: need {ctx.order_value}, have {ctx.available_cash}"
        return None


class CashReserveCheck:
    """Check 8b — post-trade cash reserve must stay above minimum."""

    def __init__(self, min_cash_reserve_pct: Decimal = Decimal("0.20")) -> None:
        self._min = min_cash_reserve_pct

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.portfolio_equity <= 0:
            return None
        post_cash = ctx.available_cash - ctx.order_value
        ratio = post_cash / ctx.portfolio_equity
        if ratio < self._min:
            return f"Post-trade cash reserve {float(ratio):.1%} below min {float(self._min):.1%}"
        return None


class StopLossRequiredCheck:
    """Check 9 — stop-loss must be set when required."""

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.require_stop_loss and ctx.stop_loss_price is None:
            return "Stop-loss price is required but not set"
        return None


class DuplicateOrderCheck:
    """Check 10 — no duplicate pending order for the same symbol."""

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.has_pending_order and ctx.symbol:
            return f"Duplicate pending order for {ctx.symbol}"
        return None


class CrossMarketExposureCheck:
    """Check 11 — cross-market exposure within configured limit."""

    def check(self, ctx: CheckContext) -> str | None:
        markets = ctx.markets_active or []
        multi_market = len(markets) > 1

        # Fail-closed: multi-market but exposure not provided
        if multi_market and ctx.cross_market_exposure_pct is None:
            return (
                "Cross-market exposure unknown: multiple markets active "
                f"({', '.join(markets)}) but cross_market_exposure_pct "
                "not provided"
            )

        # Exceed limit: check exposure pct vs max regardless of markets_active
        if (
            ctx.cross_market_exposure_pct is not None
            and ctx.max_cross_market_exposure_pct is not None
            and ctx.cross_market_exposure_pct > ctx.max_cross_market_exposure_pct
        ):
            return (
                f"Cross-market exposure "
                f"{float(ctx.cross_market_exposure_pct):.1%} "
                f"exceeds max "
                f"{float(ctx.max_cross_market_exposure_pct):.1%}"
            )
        return None


class RegimeGateCheck:
    """Check 12 — regime must allow new longs."""

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.regime_state is not None and not ctx.regime_state.allow_new_longs:
            return f"Check 12 FAIL: regime '{ctx.regime_state.regime}' blocks new longs"
        return None


class ParamFreshnessCheck:
    """Check 13 — strategy parameters must be fresh for OU/pairs strategies."""

    def check(self, ctx: CheckContext) -> str | None:
        if (
            ctx.strategy_name is not None
            and ctx.strategy_name in _PARAM_FRESHNESS_STRATEGIES
            and ctx.param_age_bars is not None
            and ctx.param_age_bars > _MAX_PARAM_AGE_BARS
        ):
            return (
                f"Check 13 FAIL: {ctx.strategy_name} params stale "
                f"({ctx.param_age_bars} bars > max {_MAX_PARAM_AGE_BARS})"
            )
        return None


class CorrelationLimitCheck:
    """Check 14 — max 3 correlated open positions (r > 0.7)."""

    def check(self, ctx: CheckContext) -> str | None:
        if ctx.open_positions is None or ctx.correlations is None or not ctx.symbol:
            return None
        from finalayze.risk.correlation import count_correlated_positions  # noqa: PLC0415

        correlated = count_correlated_positions(
            ctx.symbol,
            ctx.open_positions,
            ctx.correlations,
            threshold=_CORRELATION_THRESHOLD,
        )
        if correlated >= _MAX_CORRELATED_POSITIONS:
            return (
                f"Check 14 FAIL: {correlated} correlated positions "
                f"(max {_MAX_CORRELATED_POSITIONS})"
            )
        return None


# ── Registry ───────────────────────────────────────────────────────────────


class PreTradeChecker:
    """Runs a registry of :class:`PreTradeCheck` adapters against a context.

    Interface: one method, one typed context, one result.

        result = checker.check(ctx)

    Configuration (max sizes, thresholds) is injected at construction time.
    Pass ``checks=`` to replace the default registry with custom adapters.
    """

    def __init__(
        self,
        max_position_pct: Decimal = Decimal("0.20"),
        max_positions_per_market: int = 10,
        pdt_tracker: PDTTracker | None = None,
        max_sector_concentration_pct: Decimal = Decimal("0.40"),
        min_cash_reserve_pct: Decimal = Decimal("0.20"),
        checks: list[PreTradeCheck] | None = None,
    ) -> None:
        self._max_position_pct = max_position_pct
        self._max_positions = max_positions_per_market
        self._pdt_tracker = pdt_tracker
        self._max_sector_pct = max_sector_concentration_pct
        self._min_cash_reserve_pct = min_cash_reserve_pct
        self._checks: list[PreTradeCheck] = (
            checks if checks is not None else self._build_default_checks()
        )

    def _build_default_checks(self) -> list[PreTradeCheck]:
        return [
            MarketHoursCheck(),
            CircuitBreakerCheck(),
            PDTCheck(self._pdt_tracker),
            PositionSizeCheck(self._max_position_pct),
            MaxPositionsCheck(self._max_positions),
            SectorConcentrationCheck(self._max_sector_pct),
            CashCheck(),
            CashReserveCheck(self._min_cash_reserve_pct),
            StopLossRequiredCheck(),
            DuplicateOrderCheck(),
            CrossMarketExposureCheck(),
            RegimeGateCheck(),
            ParamFreshnessCheck(),
            CorrelationLimitCheck(),
        ]

    def check(self, ctx: CheckContext) -> PreTradeResult:
        """Run all checks against *ctx* and return the aggregated result."""
        violations: list[str] = []
        for guard in self._checks:
            v = guard.check(ctx)
            if v is not None:
                violations.append(v)
        result = PreTradeResult(passed=len(violations) == 0, violations=violations)
        if not result.passed:
            _log.warning(
                "pre_trade_check_failed",
                symbol=ctx.symbol,
                market=ctx.market_id,
                violations=result.violations,
                order_value=float(ctx.order_value),
            )
        return result

    @staticmethod
    def _is_market_open(market_id: str, dt: datetime) -> bool:
        """Kept for any direct callers."""
        return _is_market_open(market_id, dt)
