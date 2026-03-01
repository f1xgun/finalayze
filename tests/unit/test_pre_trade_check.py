"""Unit tests for PreTradeChecker risk checks."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from finalayze.risk.circuit_breaker import CircuitLevel
from finalayze.risk.pre_trade_check import PreTradeChecker

# ── Constants ──────────────────────────────────────────────────────────────
# A Wednesday during US market hours: 15:00 UTC = 11:00 ET
_MARKET_OPEN_DT = datetime(2026, 2, 25, 15, 0, tzinfo=UTC)
_PORTFOLIO_EQUITY = Decimal(100000)
_AVAILABLE_CASH = Decimal(50000)
_ORDER_VALUE = Decimal(5000)
_OPEN_POSITIONS = 3
_DEFAULT_MAX_SECTOR_PCT = Decimal("0.40")
_DEFAULT_MIN_CASH_RESERVE_PCT = Decimal("0.20")


def _base_check_kwargs() -> dict:
    """Return kwargs that pass all standard pre-trade checks."""
    return {
        "order_value": _ORDER_VALUE,
        "portfolio_equity": _PORTFOLIO_EQUITY,
        "available_cash": _AVAILABLE_CASH,
        "open_position_count": _OPEN_POSITIONS,
        "market_id": "us",
        "dt": _MARKET_OPEN_DT,
        "circuit_breaker_level": CircuitLevel.NORMAL,
    }


class TestSectorConcentration:
    """6A.2: Sector/segment concentration 40% limit."""

    def test_sector_concentration_below_limit_passes(self) -> None:
        checker = PreTradeChecker()
        kwargs = _base_check_kwargs()
        # Existing sector exposure 30k + order 5k = 35k / 100k = 35% < 40%
        kwargs["sector_exposure_value"] = Decimal(30000)
        kwargs["sector_id"] = "us_tech"
        result = checker.check(**kwargs)
        assert result.passed

    def test_sector_concentration_at_limit_fails(self) -> None:
        checker = PreTradeChecker()
        kwargs = _base_check_kwargs()
        # Existing sector exposure 36k + order 5k = 41k / 100k = 41% > 40%
        kwargs["sector_exposure_value"] = Decimal(36000)
        kwargs["sector_id"] = "us_tech"
        result = checker.check(**kwargs)
        assert not result.passed
        assert any("Sector" in v for v in result.violations)

    def test_sector_concentration_not_provided_skipped(self) -> None:
        checker = PreTradeChecker()
        kwargs = _base_check_kwargs()
        # No sector_exposure_value -> check skipped, should pass
        result = checker.check(**kwargs)
        assert result.passed

    def test_sector_concentration_custom_cap(self) -> None:
        checker = PreTradeChecker(max_sector_concentration_pct=Decimal("0.30"))
        kwargs = _base_check_kwargs()
        # 26k + 5k = 31k / 100k = 31% > 30% custom cap
        kwargs["sector_exposure_value"] = Decimal(26000)
        kwargs["sector_id"] = "us_tech"
        result = checker.check(**kwargs)
        assert not result.passed


class TestCashReserve:
    """6A.3: Min cash reserve 20% enforcement."""

    def test_cash_reserve_sufficient_passes(self) -> None:
        checker = PreTradeChecker()
        kwargs = _base_check_kwargs()
        # cash=50k, order=5k -> post_trade=45k / 100k = 45% > 20%
        result = checker.check(**kwargs)
        assert result.passed

    def test_cash_reserve_below_20pct_fails(self) -> None:
        checker = PreTradeChecker()
        kwargs = _base_check_kwargs()
        # cash=25k, order=10k -> post_trade=15k / 100k = 15% < 20%
        kwargs["available_cash"] = Decimal(25000)
        kwargs["order_value"] = Decimal(10000)
        result = checker.check(**kwargs)
        assert not result.passed
        assert any("cash reserve" in v.lower() for v in result.violations)

    def test_cash_reserve_exact_boundary(self) -> None:
        checker = PreTradeChecker()
        kwargs = _base_check_kwargs()
        # cash=25k, order=5k -> post_trade=20k / 100k = 20% == 20% (should pass)
        kwargs["available_cash"] = Decimal(25000)
        kwargs["order_value"] = Decimal(5000)
        result = checker.check(**kwargs)
        assert result.passed

    def test_cash_reserve_custom_threshold(self) -> None:
        checker = PreTradeChecker(min_cash_reserve_pct=Decimal("0.30"))
        kwargs = _base_check_kwargs()
        # cash=35k, order=10k -> post_trade=25k / 100k = 25% < 30%
        kwargs["available_cash"] = Decimal(35000)
        kwargs["order_value"] = Decimal(10000)
        result = checker.check(**kwargs)
        assert not result.passed
