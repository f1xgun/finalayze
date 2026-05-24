"""Tests for SignalExecutor private helpers extracted in Phase 2c.

Covers ``_compute_sector_exposure`` — pure arithmetic over portfolio
positions and the executor's last-price cache. ``_run_pre_trade_check``
is exercised end-to-end by existing integration tests.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.orchestration.signal_executor import SignalExecutor

# Test constants (no magic numbers per ruff PLR2004).
_QTY_A = Decimal(10)
_QTY_B = Decimal(5)
_PRICE_A = Decimal(150)
_PRICE_B = Decimal(200)


def _make_executor(last_prices: dict[str, Decimal]) -> SignalExecutor:
    executor = SignalExecutor.__new__(SignalExecutor)
    executor._last_prices = last_prices
    return executor


def _make_portfolio(positions: dict[str, Decimal]) -> MagicMock:
    p = MagicMock()
    p.positions = positions
    return p


class TestComputeSectorExposure:
    def test_returns_none_when_seg_id_empty(self) -> None:
        executor = _make_executor({"AAPL": _PRICE_A})
        portfolio = _make_portfolio({"AAPL": _QTY_A})
        assert executor._compute_sector_exposure(portfolio, seg_id="") is None

    def test_sums_qty_times_last_price_across_positions(self) -> None:
        executor = _make_executor({"AAPL": _PRICE_A, "MSFT": _PRICE_B})
        portfolio = _make_portfolio({"AAPL": _QTY_A, "MSFT": _QTY_B})
        result = executor._compute_sector_exposure(portfolio, seg_id="us_tech")
        # 10 * 150 + 5 * 200 = 1500 + 1000 = 2500
        assert result == _QTY_A * _PRICE_A + _QTY_B * _PRICE_B

    def test_skips_zero_or_negative_quantities(self) -> None:
        executor = _make_executor({"AAPL": _PRICE_A, "MSFT": _PRICE_B})
        portfolio = _make_portfolio({"AAPL": _QTY_A, "MSFT": Decimal(0)})
        result = executor._compute_sector_exposure(portfolio, seg_id="us_tech")
        assert result == _QTY_A * _PRICE_A

    def test_missing_last_price_treats_position_as_zero(self) -> None:
        # _get_last_price returns _ZERO for symbols not in cache (SIZE-02 fallback)
        executor = _make_executor({"AAPL": _PRICE_A})  # MSFT missing
        portfolio = _make_portfolio({"AAPL": _QTY_A, "MSFT": _QTY_B})
        result = executor._compute_sector_exposure(portfolio, seg_id="us_tech")
        assert result == _QTY_A * _PRICE_A

    def test_empty_portfolio_returns_zero_when_seg_id_present(self) -> None:
        executor = _make_executor({})
        portfolio = _make_portfolio({})
        assert executor._compute_sector_exposure(portfolio, seg_id="us_tech") == Decimal(0)
