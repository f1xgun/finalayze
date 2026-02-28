"""Tests for cross-market FX conversion in equity computation (5.3)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.markets.currency import CurrencyConverter


def _make_trading_loop(
    market_equities: dict[str, Decimal | None],
) -> MagicMock:
    """Create a minimal TradingLoop-like object for testing _compute_total_equity_base."""
    from finalayze.core.trading_loop import TradingLoop

    settings = MagicMock()
    settings.news_cycle_minutes = 30
    settings.strategy_cycle_minutes = 60
    settings.daily_reset_hour_utc = 0
    settings.mode = "test"
    settings.max_position_pct = 0.20
    settings.max_positions_per_market = 10
    settings.daily_loss_limit_pct = 0.05
    settings.kelly_fraction = 0.5

    loop = MagicMock(spec=TradingLoop)
    loop._fx = CurrencyConverter(base_currency="USD")
    loop._circuit_breakers = {m: MagicMock() for m in market_equities}

    def get_equity(m: str) -> Decimal | None:
        return market_equities[m]

    loop._get_market_equity = MagicMock(side_effect=get_equity)
    loop._compute_total_equity_base = TradingLoop._compute_total_equity_base.__get__(loop)
    return loop


class TestComputeTotalEquityBase:
    def test_rub_converted_to_usd(self) -> None:
        """MOEX equity (RUB) should be converted to USD before summing."""
        loop = _make_trading_loop(
            {
                "us": Decimal(10000),
                "moex": Decimal(900000),  # 900k RUB = 10k USD at rate 90
            }
        )

        total = loop._compute_total_equity_base()

        # MOEX 900k RUB ≈ 10k USD, plus US 10k USD ≈ 20k USD
        assert abs(total - Decimal(20000)) < Decimal("0.01")

    def test_single_us_market_unchanged(self) -> None:
        """Single US market equity should pass through unchanged."""
        loop = _make_trading_loop({"us": Decimal(50000)})

        total = loop._compute_total_equity_base()

        assert total == Decimal(50000)

    def test_single_moex_market_converted(self) -> None:
        """Single MOEX market equity should be converted to USD."""
        loop = _make_trading_loop({"moex": Decimal(9000)})

        total = loop._compute_total_equity_base()

        # CurrencyConverter uses 1/90.0 rate which has repeating decimals
        assert abs(total - Decimal(100)) < Decimal("0.01")

    def test_none_equity_skipped(self) -> None:
        """Markets returning None equity should be skipped."""
        loop = _make_trading_loop(
            {
                "us": Decimal(10000),
                "moex": None,
            }
        )

        total = loop._compute_total_equity_base()

        assert total == Decimal(10000)
