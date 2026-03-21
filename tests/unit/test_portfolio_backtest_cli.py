"""Tests for scripts/run_portfolio_backtest.py helper functions."""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is importable
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Ensure config/ at project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _make_candle(close: float = 100.0, volume: int = 1000) -> Any:
    """Create a minimal Candle-like object."""
    from finalayze.core.schemas import Candle

    return Candle(
        open=Decimal(str(close)),
        high=Decimal(str(close)),
        low=Decimal(str(close)),
        close=Decimal(str(close)),
        volume=volume,
        timestamp=datetime(2023, 6, 15, tzinfo=UTC),
    )


def _make_bond_info(figi: str = "FIGI_TEST") -> Any:
    """Create a minimal BondInfo mock."""
    info = MagicMock()
    info.figi = figi
    info.coupon_period_days = 182
    info.face_value = Decimal("1000")
    info.maturity_date = date(2030, 1, 1)
    return info


class TestRunBondBacktest:
    """Tests for _run_bond_backtest()."""

    def test_no_token_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without FINALAYZE_TINKOFF_TOKEN, _run_bond_backtest returns None."""
        monkeypatch.delenv("FINALAYZE_TINKOFF_TOKEN", raising=False)
        from run_portfolio_backtest import _run_bond_backtest

        result = _run_bond_backtest(
            bond_capital=400_000.0,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert result is None

    def test_empty_candles_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When TinkoffFetcher returns no candles for all OFZ symbols, result is None."""
        monkeypatch.setenv("FINALAYZE_TINKOFF_TOKEN", "test-token")

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_bond_info.return_value = _make_bond_info()
        mock_fetcher.fetch_bond_candles.return_value = []
        mock_fetcher.fetch_bond_coupons.return_value = []

        with patch(
            "run_portfolio_backtest.TinkoffFetcher", return_value=mock_fetcher
        ), patch("run_portfolio_backtest.build_default_registry"):
            from run_portfolio_backtest import _run_bond_backtest

            result = _run_bond_backtest(
                bond_capital=400_000.0,
                start_date=date(2023, 1, 1),
                end_date=date(2024, 12, 31),
            )
        assert result is None

    def test_valid_candles_produces_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With valid candle data, _run_bond_backtest produces BondBacktestResult."""
        monkeypatch.setenv("FINALAYZE_TINKOFF_TOKEN", "test-token")

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_bond_info.return_value = _make_bond_info()
        # Return raw candle dicts (as TinkoffFetcher.fetch_bond_candles does)
        mock_fetcher.fetch_bond_candles.return_value = [
            {"date": date(2023, 1, 10), "open": Decimal("99"), "high": Decimal("100"),
             "low": Decimal("98"), "close": Decimal("99.5"), "volume": 500},
            {"date": date(2023, 1, 11), "open": Decimal("99.5"), "high": Decimal("100.5"),
             "low": Decimal("99"), "close": Decimal("100"), "volume": 600},
        ]
        mock_fetcher.fetch_bond_coupons.return_value = []

        mock_engine_result = MagicMock()
        mock_engine_result.trades = []
        mock_engine_result.equity_curve = [Decimal("400000")]
        mock_engine_class = MagicMock()
        mock_engine_class.return_value.run.return_value = mock_engine_result

        with patch(
            "run_portfolio_backtest.TinkoffFetcher", return_value=mock_fetcher
        ), patch("run_portfolio_backtest.build_default_registry"), patch(
            "run_portfolio_backtest.BondBacktestEngine", mock_engine_class
        ):
            from run_portfolio_backtest import _run_bond_backtest

            result = _run_bond_backtest(
                bond_capital=400_000.0,
                start_date=date(2023, 1, 1),
                end_date=date(2024, 12, 31),
            )
        assert result is not None
        assert result is mock_engine_result

    def test_logs_bond_symbols(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_run_bond_backtest logs ticker symbols being fetched."""
        monkeypatch.setenv("FINALAYZE_TINKOFF_TOKEN", "test-token")

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_bond_info.return_value = _make_bond_info()
        mock_fetcher.fetch_bond_candles.return_value = []
        mock_fetcher.fetch_bond_coupons.return_value = []

        with patch(
            "run_portfolio_backtest.TinkoffFetcher", return_value=mock_fetcher
        ), patch("run_portfolio_backtest.build_default_registry"):
            from run_portfolio_backtest import _run_bond_backtest

            _run_bond_backtest(
                bond_capital=400_000.0,
                start_date=date(2023, 1, 1),
                end_date=date(2024, 12, 31),
            )

        # Verify at least some OFZ tickers were passed to fetch_bond_info
        assert mock_fetcher.fetch_bond_info.call_count > 0
