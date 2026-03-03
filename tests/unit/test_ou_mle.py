"""Tests for exact Ornstein-Uhlenbeck MLE fitting."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.ou_mean_reversion import (
    OUParams,
    fit_ou_exact_mle,
    fit_ou_mle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

_MIN_POINTS_FOR_MLE = 30


def _make_candle(close: float, idx: int = 0) -> Candle:
    """Build a minimal Candle for testing."""
    return Candle(
        symbol="TEST",
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=idx),
        open=Decimal(str(close)),
        high=Decimal(str(close)),
        low=Decimal(str(close)),
        close=Decimal(str(close)),
        volume=100,
    )


def _simulate_ou(
    mu: float,
    theta: float,
    sigma: float,
    dt: float,
    n: int,
    x0: float | None = None,
) -> list[float]:
    """Simulate a discrete OU process (exact transition).

    x_{t+1} = theta + (x_t - theta) * exp(-mu*dt) + sigma*sqrt((1-exp(-2*mu*dt))/(2*mu)) * Z
    """
    x0_val = x0 if x0 is not None else theta
    xs = [x0_val]
    decay = math.exp(-mu * dt)
    var = sigma**2 * (1.0 - math.exp(-2.0 * mu * dt)) / (2.0 * mu)
    std = math.sqrt(var)
    for _ in range(n - 1):
        z = float(_RNG.standard_normal())
        xs.append(theta + (xs[-1] - theta) * decay + std * z)
    return xs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFitOUExactMLE:
    """Tests for the exact discrete-time MLE function."""

    def test_fit_ou_mle_basic(self) -> None:
        """Synthetic OU process: verify parameter recovery within tolerance."""
        true_mu = 0.1
        true_theta = 5.0
        true_sigma = 0.3
        dt = 1.0
        n = 2000

        log_prices = _simulate_ou(true_mu, true_theta, true_sigma, dt, n)
        result = fit_ou_exact_mle(log_prices, dt=dt)

        assert isinstance(result, OUParams)
        # Allow 30% relative error on each parameter for a 2000-point sample
        assert abs(result.mu - true_mu) / true_mu < 0.3
        assert abs(result.theta - true_theta) / true_theta < 0.05
        assert abs(result.sigma - true_sigma) / true_sigma < 0.3

    def test_fit_ou_mle_half_life(self) -> None:
        """Half-life should equal ln(2)/mu from the fitted parameters."""
        true_mu = 0.05
        true_theta = 4.0
        true_sigma = 0.2
        dt = 1.0
        n = 1500

        log_prices = _simulate_ou(true_mu, true_theta, true_sigma, dt, n)
        result = fit_ou_exact_mle(log_prices, dt=dt)

        expected_half_life = math.log(2) / result.mu
        assert abs(result.half_life - expected_half_life) < 1e-10

    def test_fit_ou_mle_insufficient_data(self) -> None:
        """Too few data points should raise ValueError."""
        with pytest.raises(ValueError, match="at least"):
            fit_ou_exact_mle([1.0, 2.0], dt=1.0)

    def test_fit_ou_mle_non_stationary(self) -> None:
        """A random walk should yield very low mu (slow/no reversion)."""
        # Random walk: x_{t+1} = x_t + noise (mu ~ 0)
        n = 1000
        walk = [0.0]
        for _ in range(n - 1):
            walk.append(walk[-1] + float(_RNG.standard_normal()) * 0.01)

        result = fit_ou_exact_mle(walk, dt=1.0)
        # mu should be very small (close to 0) for a random walk
        assert result.mu < 0.05
        # Half-life should be very long
        assert result.half_life > 14.0  # ln(2)/0.05 ~ 13.9

    def test_strategy_with_mle_flag(self) -> None:
        """Strategy should use exact MLE when use_mle=True."""
        from finalayze.strategies.ou_mean_reversion import OUMeanReversionStrategy

        true_mu = 0.08
        true_theta = 4.6
        true_sigma = 0.25
        dt = 1.0
        n = 100

        log_prices = _simulate_ou(true_mu, true_theta, true_sigma, dt, n, x0=true_theta)

        # Build candles from log prices (exponentiate)
        candles = [_make_candle(math.exp(lp), idx=i) for i, lp in enumerate(log_prices)]

        # Push current price well below theta to trigger BUY
        low_price = math.exp(true_theta - 3.0 * true_sigma)
        candles.append(_make_candle(low_price, idx=n))

        strategy = OUMeanReversionStrategy(
            ou_window=n - 1,
            entry_threshold=1.0,
            exit_threshold=0.0,
            half_life_range=(1, 200),
            use_mle=True,
        )

        signal = strategy.generate_signal(
            symbol="TEST",
            candles=candles,
            segment_id="us_broad",
        )

        # Should produce a BUY signal since price is well below theta
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert 0.0 <= signal.confidence <= 1.0
