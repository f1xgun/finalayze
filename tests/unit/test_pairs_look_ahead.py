"""Tests for pairs strategy look-ahead bias fix (issue 5.5).

Verifies that _compute_signal() uses only historical bars (excluding the
current bar) for cointegration test, beta computation, and spread statistics.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pytest

from finalayze.core.schemas import Candle

# Constants — no magic numbers (ruff PLR2004)
N_CANDLES = 80
BASE_PRICE = 100.0
Z_ENTRY = 2.0
Z_EXIT = 0.5
MIN_CONFIDENCE = 0.0
COINT_P_PASS = 0.01
EXPECTED_HIST_LEN = N_CANDLES - 1
TOO_FEW_CANDLES = 2


def _make_candles(
    n: int,
    symbol: str,
    prices: list[float],
    market_id: str = "us",
) -> list[Candle]:
    base = datetime(2023, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            market_id=market_id,
            timeframe="1d",
            timestamp=base + timedelta(days=i),
            open=Decimal(str(round(prices[i] * 0.999, 4))),
            high=Decimal(str(round(prices[i] * 1.005, 4))),
            low=Decimal(str(round(prices[i] * 0.995, 4))),
            close=Decimal(str(round(prices[i], 4))),
            volume=1000,
        )
        for i in range(n)
    ]


def _cointegrated_prices(
    n: int = N_CANDLES,
    rng_seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Build two cointegrated price series."""
    rng = np.random.default_rng(rng_seed)
    common = rng.standard_normal(n).cumsum() + BASE_PRICE
    noise_a = rng.standard_normal(n) * 0.05
    noise_b = rng.standard_normal(n) * 0.05
    prices_a = (common + noise_a).tolist()
    prices_b = (common + noise_b).tolist()
    return prices_a, prices_b


class TestCointCalledWithHistoricalOnly:
    """Verify coint() receives n-1 length arrays (current bar excluded)."""

    def test_coint_receives_hist_arrays(self) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        prices_a, prices_b = _cointegrated_prices(N_CANDLES)
        candles_a = _make_candles(N_CANDLES, "AAPL", prices_a)
        candles_b = _make_candles(N_CANDLES, "MSFT", prices_b)

        strategy = PairsStrategy()

        with patch(
            "finalayze.strategies.pairs.coint",
            return_value=(0.0, COINT_P_PASS, None),
        ) as mock_coint:
            strategy._compute_signal(
                symbol="AAPL",
                candles_a=candles_a,
                candles_b=candles_b,
                segment_id="us_tech",
                z_entry=Z_ENTRY,
                z_exit=Z_EXIT,
            )

            mock_coint.assert_called_once()
            args = mock_coint.call_args[0]
            assert len(args[0]) == EXPECTED_HIST_LEN
            assert len(args[1]) == EXPECTED_HIST_LEN


class TestSpreadStatsExcludeCurrentBar:
    """Verify spread mean/std are computed on historical data only."""

    def test_spread_stats_use_historical_bars(self) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        prices_a, prices_b = _cointegrated_prices(N_CANDLES)
        candles_a = _make_candles(N_CANDLES, "AAPL", prices_a)
        candles_b = _make_candles(N_CANDLES, "MSFT", prices_b)

        log_a = np.log(prices_a)
        log_b = np.log(prices_b)
        log_a_hist = log_a[:-1]
        log_b_hist = log_b[:-1]

        # Compute expected beta from historical data
        cov_matrix = np.cov(log_a_hist, log_b_hist)
        expected_beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])

        # Expected spread stats from historical data
        spread_hist = log_a_hist - expected_beta * log_b_hist
        expected_mean = float(spread_hist.mean())
        expected_std = float(spread_hist.std(ddof=1))

        # Expected current spread and z-score
        current_spread = log_a[-1] - expected_beta * log_b[-1]
        expected_z = float((current_spread - expected_mean) / expected_std)

        strategy = PairsStrategy()

        with (
            patch(
                "finalayze.strategies.pairs.coint",
                return_value=(0.0, COINT_P_PASS, None),
            ),
            patch.object(
                strategy,
                "get_parameters",
                return_value={"min_confidence": MIN_CONFIDENCE},
            ),
        ):
            signal = strategy._compute_signal(
                symbol="AAPL",
                candles_a=candles_a,
                candles_b=candles_b,
                segment_id="us_tech",
                z_entry=Z_ENTRY,
                z_exit=Z_EXIT,
            )

        if signal is not None:
            assert signal.features["z_score"] == pytest.approx(round(expected_z, 4), abs=1e-6)
            assert signal.features["beta"] == pytest.approx(round(expected_beta, 4), abs=1e-6)


class TestMinCandlesGuard:
    """Verify that n < 3 returns None."""

    def test_two_candles_returns_none(self) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        prices_a = [BASE_PRICE, BASE_PRICE + 1.0]
        prices_b = [BASE_PRICE, BASE_PRICE + 0.5]
        candles_a = _make_candles(TOO_FEW_CANDLES, "AAPL", prices_a)
        candles_b = _make_candles(TOO_FEW_CANDLES, "MSFT", prices_b)

        strategy = PairsStrategy()
        result = strategy._compute_signal(
            symbol="AAPL",
            candles_a=candles_a,
            candles_b=candles_b,
            segment_id="us_tech",
            z_entry=Z_ENTRY,
            z_exit=Z_EXIT,
        )
        assert result is None

    def test_one_candle_returns_none(self) -> None:
        from finalayze.strategies.pairs import PairsStrategy

        candles_a = _make_candles(1, "AAPL", [BASE_PRICE])
        candles_b = _make_candles(1, "MSFT", [BASE_PRICE])

        strategy = PairsStrategy()
        result = strategy._compute_signal(
            symbol="AAPL",
            candles_a=candles_a,
            candles_b=candles_b,
            segment_id="us_tech",
            z_entry=Z_ENTRY,
            z_exit=Z_EXIT,
        )
        assert result is None

    def test_three_candles_proceeds(self) -> None:
        """With n=3, we get 2 historical bars -- enough to proceed past guard."""
        from finalayze.strategies.pairs import PairsStrategy

        n = 3
        prices_a = [BASE_PRICE, BASE_PRICE + 1.0, BASE_PRICE + 2.0]
        prices_b = [BASE_PRICE, BASE_PRICE + 0.5, BASE_PRICE + 1.0]
        candles_a = _make_candles(n, "AAPL", prices_a)
        candles_b = _make_candles(n, "MSFT", prices_b)

        strategy = PairsStrategy()

        # coint will be called (not blocked by guard), mock it to pass
        with patch(
            "finalayze.strategies.pairs.coint",
            return_value=(0.0, COINT_P_PASS, None),
        ) as mock_coint:
            strategy._compute_signal(
                symbol="AAPL",
                candles_a=candles_a,
                candles_b=candles_b,
                segment_id="us_tech",
                z_entry=Z_ENTRY,
                z_exit=Z_EXIT,
            )
            # The key assertion: coint WAS called, meaning we got past the guard
            mock_coint.assert_called_once()
