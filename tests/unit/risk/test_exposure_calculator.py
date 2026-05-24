"""Tests for ExposureCalculator — cross-market exposure ratio computation.

Extracted from signal_executor.process_instrument (Phase 2b). Verifies:
- Sums invested value (equity - cash) across known markets in base currency.
- Adds prospective order value (also converted to base).
- Returns the ratio; zero total equity yields zero exposure.
- max_exposure_pct falls back to 0.80 when settings missing/malformed.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.risk.exposure import ExposureCalculator

# Test constants (no magic numbers per ruff PLR2004).
_US_EQUITY = Decimal(50000)
_US_CASH = Decimal(20000)
_MOEX_EQUITY = Decimal(2700000)  # ≈ $30k at 90 RUB/USD
_MOEX_CASH = Decimal(900000)  # ≈ $10k
_ORDER_VALUE_USD = Decimal(5000)
_USDRUB = Decimal(90)


def _make_portfolio(*, equity: Decimal, cash: Decimal) -> MagicMock:
    p = MagicMock()
    p.equity = equity
    p.cash = cash
    return p


def _make_broker_router(market_to_portfolio: dict[str, MagicMock]) -> MagicMock:
    router = MagicMock()

    def route(market_id: str) -> MagicMock:
        broker = MagicMock()
        broker.get_portfolio.return_value = market_to_portfolio.get(market_id)
        return broker

    router.route.side_effect = route
    return router


def _market_equity(market_id: str, portfolios: dict[str, MagicMock]) -> Decimal | None:
    p = portfolios.get(market_id)
    return Decimal(str(p.equity)) if p else None


class TestSingleMarket:
    def test_us_only_zero_invested_yields_order_ratio(self) -> None:
        portfolios = {"us": _make_portfolio(equity=_US_EQUITY, cash=_US_EQUITY)}
        calc = ExposureCalculator(
            broker_router=_make_broker_router(portfolios),
            symbol_limit_markets=["us"],
            settings=MagicMock(max_cross_market_exposure_pct=0.80),
            get_market_equity=lambda m: _market_equity(m, portfolios),
        )
        # No existing invested; only the prospective order counts.
        cross, max_pct = calc.compute(market_id="us", order_value=_ORDER_VALUE_USD)
        assert cross == _ORDER_VALUE_USD / _US_EQUITY
        assert max_pct == Decimal("0.80")

    def test_partially_invested_us(self) -> None:
        portfolios = {"us": _make_portfolio(equity=_US_EQUITY, cash=_US_CASH)}
        calc = ExposureCalculator(
            broker_router=_make_broker_router(portfolios),
            symbol_limit_markets=["us"],
            settings=MagicMock(max_cross_market_exposure_pct=0.80),
            get_market_equity=lambda m: _market_equity(m, portfolios),
        )
        # Invested = 50000 - 20000 = 30000; prospective = 5000 ⇒ 35000 / 50000 = 0.70
        cross, _ = calc.compute(market_id="us", order_value=_ORDER_VALUE_USD)
        assert cross == Decimal("0.70")


class TestMultiMarketFX:
    def test_moex_invested_converted_to_usd(self) -> None:
        portfolios = {
            "us": _make_portfolio(equity=_US_EQUITY, cash=_US_CASH),
            "moex": _make_portfolio(equity=_MOEX_EQUITY, cash=_MOEX_CASH),
        }
        # MOEX invested in RUB: 2_700_000 - 900_000 = 1_800_000 RUB ≈ 20_000 USD
        # US invested in USD: 50_000 - 20_000 = 30_000 USD
        # Total invested: 50_000 USD
        # Total equity: 50_000 (US) + 30_000 (MOEX/USD) = 80_000 USD
        # Order: 5_000 USD on US ⇒ prospective = 55_000 / 80_000 = 0.6875
        calc = ExposureCalculator(
            broker_router=_make_broker_router(portfolios),
            symbol_limit_markets=["us", "moex"],
            settings=MagicMock(max_cross_market_exposure_pct=0.80),
            get_market_equity=lambda m: _market_equity(m, portfolios),
        )
        cross, _ = calc.compute(market_id="us", order_value=_ORDER_VALUE_USD)
        assert cross == Decimal(55000) / Decimal(80000)


class TestEdgeCases:
    def test_zero_total_equity_returns_zero_exposure(self) -> None:
        calc = ExposureCalculator(
            broker_router=MagicMock(),
            symbol_limit_markets=[],
            settings=MagicMock(max_cross_market_exposure_pct=0.80),
            get_market_equity=lambda _: None,  # all markets unknown
        )
        cross, _ = calc.compute(market_id="us", order_value=_ORDER_VALUE_USD)
        assert cross == Decimal(0)

    def test_max_exposure_fallback_on_invalid_setting(self) -> None:
        settings = MagicMock()
        settings.max_cross_market_exposure_pct = "not a number"
        portfolios = {"us": _make_portfolio(equity=_US_EQUITY, cash=_US_EQUITY)}
        calc = ExposureCalculator(
            broker_router=_make_broker_router(portfolios),
            symbol_limit_markets=["us"],
            settings=settings,
            get_market_equity=lambda m: _market_equity(m, portfolios),
        )
        _, max_pct = calc.compute(market_id="us", order_value=_ORDER_VALUE_USD)
        assert max_pct == Decimal("0.80")

    def test_unknown_market_currency_defaults_to_usd(self) -> None:
        portfolios = {"unknown": _make_portfolio(equity=Decimal(100), cash=Decimal(50))}
        calc = ExposureCalculator(
            broker_router=_make_broker_router(portfolios),
            symbol_limit_markets=["unknown"],
            settings=MagicMock(max_cross_market_exposure_pct=0.80),
            get_market_equity=lambda m: _market_equity(m, portfolios),
        )
        # Unknown market is treated as USD; 50 invested + 5000 order = 5050 / 100 = 50.5
        cross, _ = calc.compute(market_id="unknown", order_value=_ORDER_VALUE_USD)
        assert cross == Decimal(5050) / Decimal(100)
