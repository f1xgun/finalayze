"""RED tests for CheckContext + PreTradeCheck registry refactor.

Verifies:
  1. CheckContext is a frozen dataclass replacing the 22-param check() signature.
  2. PreTradeCheck is a Protocol: any class with check(ctx) -> str | None qualifies.
  3. PreTradeChecker.check(ctx: CheckContext) is the new single-entry interface.
  4. Each built-in check class can be instantiated and exercised in isolation.
  5. MarketHoursCheck, CircuitBreakerCheck, PDTCheck, PositionSizeCheck,
     MaxPositionsCheck, SectorConcentrationCheck, CashCheck, CashReserveCheck,
     StopLossRequiredCheck, DuplicateOrderCheck, CrossMarketExposureCheck,
     RegimeGateCheck, ParamFreshnessCheck, CorrelationLimitCheck all live in
     finalayze.risk.pre_trade_check.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from finalayze.risk.circuit_breaker import CircuitLevel
from finalayze.risk.pre_trade_check import (
    CheckContext,
    CircuitBreakerCheck,
    CorrelationLimitCheck,
    CrossMarketExposureCheck,
    DuplicateOrderCheck,
    MarketHoursCheck,
    MaxPositionsCheck,
    ParamFreshnessCheck,
    PDTCheck,
    PDTTracker,
    PositionSizeCheck,
    PreTradeCheck,
    PreTradeChecker,
    RegimeGateCheck,
    SectorConcentrationCheck,
    StopLossRequiredCheck,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

_MARKET_OPEN_DT = datetime(2026, 2, 25, 15, 0, tzinfo=UTC)  # Wednesday, 11:00 ET
_EQUITY = Decimal(100_000)
_CASH = Decimal(50_000)
_ORDER = Decimal(5_000)


def _ctx(**overrides: object) -> CheckContext:
    """Return a CheckContext that passes all standard checks."""
    defaults: dict[str, object] = {
        "order_value": _ORDER,
        "portfolio_equity": _EQUITY,
        "available_cash": _CASH,
        "open_position_count": 3,
        "market_id": "us",
        "dt": _MARKET_OPEN_DT,
        "circuit_breaker_level": CircuitLevel.NORMAL,
    }
    defaults.update(overrides)
    return CheckContext(**defaults)  # type: ignore[arg-type]


# ── 1. CheckContext dataclass ───────────────────────────────────────────────


class TestCheckContext:
    def test_construction_with_required_fields(self) -> None:
        ctx = CheckContext(
            order_value=_ORDER,
            portfolio_equity=_EQUITY,
            available_cash=_CASH,
            open_position_count=3,
        )
        assert ctx.order_value == _ORDER
        assert ctx.portfolio_equity == _EQUITY
        assert ctx.available_cash == _CASH
        assert ctx.open_position_count == 3
        assert ctx.market_id == "us"  # default

    def test_frozen(self) -> None:
        ctx = CheckContext(
            order_value=_ORDER,
            portfolio_equity=_EQUITY,
            available_cash=_CASH,
            open_position_count=0,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            ctx.order_value = Decimal(1)  # type: ignore[misc]

    def test_all_optional_fields_default_none_or_falsy(self) -> None:
        ctx = CheckContext(
            order_value=_ORDER,
            portfolio_equity=_EQUITY,
            available_cash=_CASH,
            open_position_count=0,
        )
        assert ctx.circuit_breaker_level is None
        assert ctx.regime_state is None
        assert ctx.stop_loss_price is None
        assert ctx.markets_active is None
        assert not ctx.require_stop_loss
        assert not ctx.has_pending_order
        assert not ctx.is_day_trade


# ── 2. PreTradeCheck protocol ───────────────────────────────────────────────


class TestPreTradeCheckProtocol:
    def test_custom_check_satisfies_protocol(self) -> None:
        class AlwaysBlock:
            def check(self, ctx: CheckContext) -> str | None:
                return "blocked"

        checker = PreTradeChecker(checks=[AlwaysBlock()])
        result = checker.check(_ctx())
        assert not result.passed
        assert "blocked" in result.violations

    def test_custom_check_returning_none_passes(self) -> None:
        class NeverBlock:
            def check(self, ctx: CheckContext) -> str | None:
                return None

        checker = PreTradeChecker(checks=[NeverBlock()])
        result = checker.check(_ctx())
        assert result.passed

    def test_pre_trade_check_is_runtime_checkable(self) -> None:
        """Protocol import succeeds and has the expected attribute."""
        assert hasattr(PreTradeCheck, "__protocol_attrs__") or callable(PreTradeCheck)


# ── 3. PreTradeChecker.check(ctx) — new entry point ────────────────────────


class TestPreTradeCheckerNewInterface:
    def test_passing_context_returns_passed_result(self) -> None:
        checker = PreTradeChecker()
        result = checker.check(_ctx())
        assert result.passed

    def test_result_has_no_violations_on_pass(self) -> None:
        checker = PreTradeChecker()
        result = checker.check(_ctx())
        assert result.violations == []

    def test_failing_context_populates_violations(self) -> None:
        checker = PreTradeChecker()
        ctx = _ctx(order_value=_EQUITY * 2)  # way too large
        result = checker.check(ctx)
        assert not result.passed
        assert len(result.violations) >= 1


# ── 4. Individual check classes ─────────────────────────────────────────────


class TestMarketHoursCheck:
    def test_open_passes(self) -> None:
        assert MarketHoursCheck().check(_ctx(market_id="us", dt=_MARKET_OPEN_DT)) is None

    def test_closed_fails(self) -> None:
        closed = datetime(2026, 2, 25, 1, 0, tzinfo=UTC)  # midnight UTC
        violation = MarketHoursCheck().check(_ctx(market_id="us", dt=closed))
        assert violation is not None
        assert "closed" in violation.lower() or "market" in violation.lower()


class TestCircuitBreakerCheck:
    def test_normal_passes(self) -> None:
        assert CircuitBreakerCheck().check(_ctx(circuit_breaker_level=CircuitLevel.NORMAL)) is None

    def test_halted_fails(self) -> None:
        violation = CircuitBreakerCheck().check(_ctx(circuit_breaker_level=CircuitLevel.HALTED))
        assert violation is not None

    def test_none_level_passes(self) -> None:
        assert CircuitBreakerCheck().check(_ctx(circuit_breaker_level=None)) is None


class TestPDTCheck:
    def test_non_us_market_never_fails(self) -> None:
        tracker = PDTTracker()
        for _ in range(5):
            tracker.record_day_trade(_MARKET_OPEN_DT.date())
        check = PDTCheck(tracker)
        violation = check.check(
            _ctx(market_id="moex", is_day_trade=True, portfolio_equity=Decimal(5_000))
        )
        assert violation is None

    def test_us_equity_above_threshold_exempt(self) -> None:
        tracker = PDTTracker()
        for _ in range(5):
            tracker.record_day_trade(_MARKET_OPEN_DT.date())
        check = PDTCheck(tracker)
        violation = check.check(
            _ctx(market_id="us", is_day_trade=True, portfolio_equity=Decimal(30_000))
        )
        assert violation is None

    def test_us_too_many_day_trades_fails(self) -> None:
        tracker = PDTTracker()
        for _ in range(3):
            tracker.record_day_trade(_MARKET_OPEN_DT.date())
        check = PDTCheck(tracker)
        violation = check.check(
            _ctx(market_id="us", is_day_trade=True, portfolio_equity=Decimal(5_000))
        )
        assert violation is not None


class TestPositionSizeCheck:
    def test_within_limit_passes(self) -> None:
        # 5% order on 100k equity with 20% max → passes
        assert PositionSizeCheck(Decimal("0.20")).check(_ctx()) is None

    def test_exceeds_limit_fails(self) -> None:
        violation = PositionSizeCheck(Decimal("0.04")).check(_ctx())  # 5% > 4%
        assert violation is not None
        assert "size" in violation.lower() or "exceed" in violation.lower()

    def test_zero_equity_fails(self) -> None:
        violation = PositionSizeCheck(Decimal("0.20")).check(_ctx(portfolio_equity=Decimal(0)))
        assert violation is not None


class TestMaxPositionsCheck:
    def test_below_max_passes(self) -> None:
        assert MaxPositionsCheck(10).check(_ctx(open_position_count=3)) is None

    def test_at_max_fails(self) -> None:
        violation = MaxPositionsCheck(3).check(_ctx(open_position_count=3))
        assert violation is not None


class TestSectorConcentrationCheck:
    def test_no_sector_data_skipped(self) -> None:
        ctx = _ctx()  # no sector_exposure_value, no sector_id
        assert SectorConcentrationCheck(Decimal("0.40")).check(ctx) is None

    def test_within_limit_passes(self) -> None:
        ctx = _ctx(sector_exposure_value=Decimal(30_000), sector_id="us_tech")
        assert SectorConcentrationCheck(Decimal("0.40")).check(ctx) is None

    def test_exceeds_limit_fails(self) -> None:
        ctx = _ctx(sector_exposure_value=Decimal(36_000), sector_id="us_tech")
        violation = SectorConcentrationCheck(Decimal("0.40")).check(ctx)
        assert violation is not None


class TestStopLossRequiredCheck:
    def test_not_required_passes(self) -> None:
        ctx = _ctx(require_stop_loss=False, stop_loss_price=None)
        assert StopLossRequiredCheck().check(ctx) is None

    def test_required_and_set_passes(self) -> None:
        ctx = _ctx(require_stop_loss=True, stop_loss_price=Decimal(99))
        assert StopLossRequiredCheck().check(ctx) is None

    def test_required_but_missing_fails(self) -> None:
        ctx = _ctx(require_stop_loss=True, stop_loss_price=None)
        violation = StopLossRequiredCheck().check(ctx)
        assert violation is not None


class TestDuplicateOrderCheck:
    def test_no_pending_passes(self) -> None:
        ctx = _ctx(has_pending_order=False, symbol="AAPL")
        assert DuplicateOrderCheck().check(ctx) is None

    def test_pending_order_fails(self) -> None:
        ctx = _ctx(has_pending_order=True, symbol="AAPL")
        violation = DuplicateOrderCheck().check(ctx)
        assert violation is not None
        assert "AAPL" in violation


class TestCrossMarketExposureCheck:
    def test_single_market_no_exposure_skips_failclosed(self) -> None:
        ctx = _ctx(markets_active=["us"], cross_market_exposure_pct=None)
        assert CrossMarketExposureCheck().check(ctx) is None

    def test_single_market_exceeds_pct_still_fails(self) -> None:
        ctx = _ctx(
            markets_active=["us"],
            cross_market_exposure_pct=Decimal("0.90"),
            max_cross_market_exposure_pct=Decimal("0.40"),
        )
        violation = CrossMarketExposureCheck().check(ctx)
        assert violation is not None

    def test_multi_market_no_exposure_provided_fails(self) -> None:
        ctx = _ctx(markets_active=["us", "moex"], cross_market_exposure_pct=None)
        violation = CrossMarketExposureCheck().check(ctx)
        assert violation is not None

    def test_within_limit_passes(self) -> None:
        ctx = _ctx(
            markets_active=["us", "moex"],
            cross_market_exposure_pct=Decimal("0.30"),
            max_cross_market_exposure_pct=Decimal("0.40"),
        )
        assert CrossMarketExposureCheck().check(ctx) is None

    def test_exceeds_limit_fails(self) -> None:
        ctx = _ctx(
            markets_active=["us", "moex"],
            cross_market_exposure_pct=Decimal("0.50"),
            max_cross_market_exposure_pct=Decimal("0.40"),
        )
        violation = CrossMarketExposureCheck().check(ctx)
        assert violation is not None


class TestRegimeGateCheck:
    def test_no_regime_state_passes(self) -> None:
        assert RegimeGateCheck().check(_ctx(regime_state=None)) is None

    def test_allows_new_longs_passes(self) -> None:
        regime = SimpleNamespace(allow_new_longs=True, regime="trending_up")
        assert RegimeGateCheck().check(_ctx(regime_state=regime)) is None

    def test_blocks_new_longs_fails(self) -> None:
        regime = SimpleNamespace(allow_new_longs=False, regime="bear")
        violation = RegimeGateCheck().check(_ctx(regime_state=regime))
        assert violation is not None


class TestParamFreshnessCheck:
    def test_non_freshness_strategy_skipped(self) -> None:
        ctx = _ctx(strategy_name="momentum", param_age_bars=100)
        assert ParamFreshnessCheck().check(ctx) is None

    def test_fresh_params_passes(self) -> None:
        ctx = _ctx(strategy_name="ou_mean_reversion", param_age_bars=3)
        assert ParamFreshnessCheck().check(ctx) is None

    def test_stale_params_fails(self) -> None:
        ctx = _ctx(strategy_name="ou_mean_reversion", param_age_bars=10)
        violation = ParamFreshnessCheck().check(ctx)
        assert violation is not None


class TestCorrelationLimitCheck:
    def test_no_correlation_data_skipped(self) -> None:
        ctx = _ctx(symbol="AAPL", open_positions=None, correlations=None)
        assert CorrelationLimitCheck().check(ctx) is None

    def test_below_limit_passes(self) -> None:
        ctx = _ctx(
            symbol="AAPL",
            open_positions=["MSFT"],
            correlations={("AAPL", "MSFT"): 0.8},
        )
        assert CorrelationLimitCheck().check(ctx) is None

    def test_at_limit_fails(self) -> None:
        ctx = _ctx(
            symbol="AAPL",
            open_positions=["MSFT", "GOOGL", "META"],
            correlations={
                ("AAPL", "MSFT"): 0.9,
                ("AAPL", "GOOGL"): 0.85,
                ("AAPL", "META"): 0.75,
            },
        )
        violation = CorrelationLimitCheck().check(ctx)
        assert violation is not None
