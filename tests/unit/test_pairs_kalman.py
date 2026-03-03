"""Tests for Kalman filter hedge ratio in PairsStrategy (Layer 4)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from finalayze.core.exceptions import InsufficientDataError
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.pairs import PairsStrategy, compute_kalman_hedge_ratio

# ---------------------------------------------------------------------------
# Constants (avoid ruff PLR2004)
# ---------------------------------------------------------------------------
_MIN_KALMAN_POINTS = 20
_SERIES_LEN_SHORT = 10
_SERIES_LEN_LONG = 100
_TRUE_BETA = 1.5
_TRUE_ALPHA = 0.1
_NOISE_STD = 0.01
_REGIME_CHANGE_BETA = 0.5
_REGIME_SPLIT = 50
_REGIME_TOTAL = 100
_Z_ENTRY = 2.0
_Z_EXIT = 0.5
_MIN_CONFIDENCE = 0.4
_NUM_CANDLES = 80
_BASE_PRICE_A = 100.0
_BASE_PRICE_B = 50.0
_SEED = 42
_BETA_TOLERANCE = 0.3
_ALPHA_TOLERANCE = 0.5
_REGIME2_BETA_TOLERANCE = 0.4


def _make_candle(
    symbol: str,
    price: float,
    idx: int,
    market_id: str = "US",
) -> Candle:
    """Helper to build a Candle with a unique timestamp."""
    ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=idx)
    return Candle(
        symbol=symbol,
        market_id=market_id,
        timeframe="1h",
        timestamp=ts,
        open=Decimal(str(price)),
        high=Decimal(str(price * 1.01)),
        low=Decimal(str(price * 0.99)),
        close=Decimal(str(price)),
        volume=1000,
    )


class TestComputeKalmanHedgeRatio:
    """Tests for the compute_kalman_hedge_ratio function."""

    def test_kalman_hedge_ratio_basic(self) -> None:
        """Known cointegrated series: y = alpha + beta * x + noise."""
        rng = np.random.default_rng(_SEED)
        x_prices = np.cumsum(rng.standard_normal(_SERIES_LEN_LONG)) + _BASE_PRICE_A
        y_prices = _TRUE_ALPHA + _TRUE_BETA * x_prices + rng.normal(0, _NOISE_STD, _SERIES_LEN_LONG)

        alpha, beta = compute_kalman_hedge_ratio(y_prices.tolist(), x_prices.tolist())

        # Kalman filter should converge close to the true parameters
        assert abs(beta - _TRUE_BETA) < _BETA_TOLERANCE, f"beta={beta:.4f}, expected ~{_TRUE_BETA}"
        # Alpha may not converge as tightly, but should be in the right ballpark
        assert abs(alpha - _TRUE_ALPHA) < _ALPHA_TOLERANCE, (
            f"alpha={alpha:.4f}, expected ~{_TRUE_ALPHA}"
        )

    def test_kalman_hedge_ratio_insufficient_data(self) -> None:
        """Should raise InsufficientDataError with fewer than 20 data points."""
        y_prices = list(range(_SERIES_LEN_SHORT))
        x_prices = list(range(_SERIES_LEN_SHORT))

        with pytest.raises(InsufficientDataError):
            compute_kalman_hedge_ratio(
                [float(y) for y in y_prices],
                [float(x) for x in x_prices],
            )

    def test_kalman_adapts_to_regime_change(self) -> None:
        """Beta shifts when the underlying relationship changes mid-series."""
        rng = np.random.default_rng(_SEED)
        x_prices = np.cumsum(rng.standard_normal(_REGIME_TOTAL)) + _BASE_PRICE_A

        # Regime 1: beta = TRUE_BETA, Regime 2: beta = REGIME_CHANGE_BETA
        y_regime1 = _TRUE_BETA * x_prices[:_REGIME_SPLIT] + rng.normal(0, _NOISE_STD, _REGIME_SPLIT)
        y_regime2 = _REGIME_CHANGE_BETA * x_prices[_REGIME_SPLIT:] + rng.normal(
            0, _NOISE_STD, _REGIME_SPLIT
        )
        y_prices = np.concatenate([y_regime1, y_regime2])

        # Run Kalman on the full series -- final beta should be closer to regime 2
        _, beta = compute_kalman_hedge_ratio(y_prices.tolist(), x_prices.tolist())

        assert abs(beta - _REGIME_CHANGE_BETA) < _REGIME2_BETA_TOLERANCE, (
            f"beta={beta:.4f}, expected ~{_REGIME_CHANGE_BETA} after regime change"
        )


class TestPairsStrategyWithKalman:
    """Integration test: PairsStrategy with use_kalman flag."""

    def test_pairs_strategy_with_kalman_flag(self) -> None:
        """Strategy should use Kalman hedge ratio when use_kalman=True."""
        rng = np.random.default_rng(_SEED)
        strategy = PairsStrategy()

        # Build cointegrated candle series
        x_base = np.cumsum(rng.standard_normal(_NUM_CANDLES)) + _BASE_PRICE_A
        y_base = _TRUE_BETA * x_base + rng.normal(0, _NOISE_STD, _NUM_CANDLES)
        # Make the last bar diverge to create a z-score signal
        y_base[-1] = y_base[-1] + 10.0  # large positive deviation

        candles_a = [_make_candle("SYM_A", float(y_base[i]), i) for i in range(_NUM_CANDLES)]
        candles_b = [_make_candle("SYM_B", float(x_base[i]), i) for i in range(_NUM_CANDLES)]

        strategy.set_peer_candles("SYM_B", candles_b)

        # Monkey-patch get_parameters to return use_kalman=True without YAML
        def mock_params(segment_id: str) -> dict[str, object]:
            return {
                "pairs": [["SYM_A", "SYM_B"]],
                "z_entry": _Z_ENTRY,
                "z_exit": _Z_EXIT,
                "min_confidence": _MIN_CONFIDENCE,
                "use_kalman": True,
            }

        strategy.get_parameters = mock_params  # type: ignore[assignment]

        signal = strategy.generate_signal("SYM_A", candles_a, "test_segment")

        # We expect a signal to be generated (the divergence should trigger it)
        assert signal is not None
        assert isinstance(signal, Signal)
        assert signal.direction in (SignalDirection.BUY, SignalDirection.SELL)
        # Features should contain kalman-related info
        assert "beta" in signal.features
        assert signal.features.get("kalman") == 1.0
