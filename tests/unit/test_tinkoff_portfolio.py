"""Unit tests for TinkoffBroker.get_portfolio() cash calculation.

Verifies that currency positions are summed as cash, share positions
are excluded from cash, and missing currency positions result in zero cash.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from finalayze.execution.tinkoff_broker import TinkoffBroker
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry

SBER_FIGI = "BBG004730N88"


def _make_broker() -> TinkoffBroker:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return TinkoffBroker(token="fake_token", registry=registry, sandbox=True)  # noqa: S106


def _make_position(
    figi: str,
    units: int,
    nano: int = 0,
    instrument_type: str = "share",
) -> MagicMock:
    pos = MagicMock()
    pos.figi = figi
    pos.quantity.units = units
    pos.quantity.nano = nano
    pos.instrument_type = instrument_type
    return pos


def _make_portfolio(
    total_units: int,
    total_nano: int,
    positions: list[MagicMock],
) -> MagicMock:
    portfolio = MagicMock()
    portfolio.total_amount_portfolio.units = total_units
    portfolio.total_amount_portfolio.nano = total_nano
    portfolio.positions = positions
    return portfolio


class TestGetPortfolioCash:
    """Tests for correct cash extraction from currency positions."""

    def test_currency_positions_summed_as_cash(self) -> None:
        """Currency positions should be summed and reported as cash."""
        rub_pos = _make_position("RUB000UTSTOM", units=100_000, instrument_type="currency")
        usd_pos = _make_position("USD000UTSTOM", units=5_000, instrument_type="currency")
        total_equity = 200_000

        mock_portfolio = _make_portfolio(total_equity, 0, [rub_pos, usd_pos])

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_portfolio,
        ):
            broker = _make_broker()
            portfolio = broker.get_portfolio()

        expected_cash = Decimal(100_000) + Decimal(5_000)
        assert portfolio.cash == expected_cash
        assert portfolio.equity == Decimal(total_equity)
        # Currency positions must not appear in the positions map
        assert "RUB000UTSTOM" not in portfolio.positions
        assert "USD000UTSTOM" not in portfolio.positions

    def test_share_positions_excluded_from_cash(self) -> None:
        """Share positions should go into positions map, not cash."""
        share_pos = _make_position(SBER_FIGI, units=20, instrument_type="share")
        currency_pos = _make_position("RUB000UTSTOM", units=50_000, instrument_type="currency")
        total_equity = 100_000

        mock_portfolio = _make_portfolio(total_equity, 0, [share_pos, currency_pos])

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_portfolio,
        ):
            broker = _make_broker()
            portfolio = broker.get_portfolio()

        assert portfolio.cash == Decimal(50_000)
        assert portfolio.positions[SBER_FIGI] == Decimal(20)
        assert portfolio.equity == Decimal(total_equity)

    def test_no_currency_positions_cash_zero(self) -> None:
        """When there are no currency positions, cash should be zero."""
        share_pos = _make_position(SBER_FIGI, units=10, instrument_type="share")
        total_equity = 80_000

        mock_portfolio = _make_portfolio(total_equity, 0, [share_pos])

        with patch(
            "finalayze.execution.tinkoff_broker.asyncio.run",
            return_value=mock_portfolio,
        ):
            broker = _make_broker()
            portfolio = broker.get_portfolio()

        assert portfolio.cash == Decimal(0)
        assert portfolio.positions[SBER_FIGI] == Decimal(10)
        assert portfolio.equity == Decimal(total_equity)
