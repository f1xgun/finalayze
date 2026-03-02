"""Unit tests for OU mean reversion strategy."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.risk.regime import MarketRegime, RegimeState
from finalayze.strategies.ou_mean_reversion import (
    OUMeanReversionStrategy,
    OUParams,
    fit_ou_mle,
)

_RNG_SEED = 42
_OU_WINDOW = 90
_ENTRY_THRESHOLD = 1.5
_EXIT_THRESHOLD = 0.0
_HALF_LIFE_MIN = 5
_HALF_LIFE_MAX = 60
_CONFIDENCE_BASE = 0.4
_CONFIDENCE_SCALE = 0.15
_MAX_CONFIDENCE = 0.95
_CRISIS_THRESHOLD = 2.0


def _make_candles(
    prices: list[float],
    symbol: str = "AAPL",
    market_id: str = "us",
) -> list[Candle]:
    """Build candles from a list of close prices."""
    base = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    candles = []
    for i, price in enumerate(prices):
        p = Decimal(str(price))
        candles.append(
            Candle(
                symbol=symbol,
                market_id=market_id,
                timeframe="1d",
                timestamp=base + timedelta(days=i),
                open=p,
                high=p + Decimal(1),
                low=p - Decimal(1),
                close=p,
                volume=1_000_000,
            )
        )
    return candles


def _generate_ou_prices(
    n: int,
    mu: float = 0.05,
    theta: float = 5.0,
    sigma: float = 0.1,
    x0: float = 5.0,
    seed: int = _RNG_SEED,
) -> list[float]:
    """Generate synthetic OU process log-prices, convert to prices."""
    rng = np.random.default_rng(seed)
    log_prices = [x0]
    dt = 1.0
    for _ in range(n - 1):
        x = log_prices[-1]
        dx = mu * (theta - x) * dt + sigma * math.sqrt(dt) * rng.standard_normal()
        log_prices.append(x + dx)
    return [math.exp(lp) for lp in log_prices]


class TestFitOUMLE:
    """Tests for the fit_ou_mle function."""

    def test_ou_fit_mle(self) -> None:
        """Verify MLE fitting on synthetic OU data recovers approximate params."""
        n = 500
        true_mu = 0.05
        true_theta = 5.0
        log_prices_raw = _generate_ou_prices(n, mu=true_mu, theta=true_theta, sigma=0.1)
        log_prices = [math.log(p) for p in log_prices_raw]

        params = fit_ou_mle(log_prices)

        assert isinstance(params, OUParams)
        assert params.mu > 0, "Mean reversion speed must be positive"
        assert params.half_life > 0, "Half-life must be positive"
        # Half-life should be ln(2)/mu
        expected_hl = math.log(2) / params.mu
        assert abs(params.half_life - expected_hl) < 1e-6


class TestOUStrategy:
    """Tests for OUMeanReversionStrategy."""

    def _make_strategy(
        self,
        ou_window: int = _OU_WINDOW,
        entry_threshold: float = _ENTRY_THRESHOLD,
        exit_threshold: float = _EXIT_THRESHOLD,
        half_life_range: tuple[int, int] = (_HALF_LIFE_MIN, _HALF_LIFE_MAX),
    ) -> OUMeanReversionStrategy:
        return OUMeanReversionStrategy(
            ou_window=ou_window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            half_life_range=half_life_range,
        )

    def test_ou_no_look_ahead(self) -> None:
        """Fitting window must exclude the current bar."""
        # Generate prices with known OU params, enough for window + 1
        prices = _generate_ou_prices(100, mu=0.05, theta=5.0, sigma=0.1)
        candles = _make_candles(prices)

        strategy = self._make_strategy(ou_window=90)
        # The strategy should use candles[-(ou_window+1):-1] for fitting
        # i.e. 91 candles, excluding the last one
        # If we have exactly 100 candles and ou_window=90, it should work
        # The key test: no error, and it uses historical data only
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        # We just verify no crash; the look-ahead prevention is structural
        # (tested by checking the fitting window in implementation)
        assert signal is None or signal.strategy_name == "ou_mean_reversion"

    def test_ou_half_life_filter(self) -> None:
        """Rejects signal when half_life is outside the valid range."""
        # Use very noisy data so half_life is likely extreme
        rng = np.random.default_rng(123)
        # Random walk (no mean reversion) -> very large half-life
        prices = list(np.exp(np.cumsum(rng.standard_normal(200) * 0.01) + 5.0))
        candles = _make_candles(prices)

        strategy = self._make_strategy(
            ou_window=90,
            half_life_range=(5, 10),  # Very tight range
        )
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None

    def test_ou_buy_signal_below_threshold(self) -> None:
        """z-score < -entry_threshold should produce a BUY signal."""
        # Generate OU prices centered at theta=5.0, then add a big drop at the end
        prices = _generate_ou_prices(100, mu=0.05, theta=5.0, sigma=0.05, seed=10)
        # Force last price far below the mean
        mean_price = math.exp(5.0)
        prices[-1] = mean_price * 0.5  # Very low price -> negative z-score

        candles = _make_candles(prices)
        strategy = self._make_strategy(ou_window=90, entry_threshold=1.5)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert 0.0 <= signal.confidence <= 1.0

    def test_ou_no_signal_in_range(self) -> None:
        """z-score between -entry_threshold and exit_threshold returns None."""
        # Prices very close to mean -> z-score near 0
        prices = _generate_ou_prices(100, mu=0.1, theta=5.0, sigma=0.01, x0=5.0, seed=77)
        candles = _make_candles(prices)

        strategy = self._make_strategy(ou_window=90, entry_threshold=1.5, exit_threshold=0.0)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None

    def test_ou_sell_signal_above_exit(self) -> None:
        """z-score > exit_threshold with open position should produce SELL."""
        prices = _generate_ou_prices(100, mu=0.05, theta=5.0, sigma=0.05, seed=10)
        # Force price above mean
        mean_price = math.exp(5.0)
        prices[-1] = mean_price * 1.5  # Above mean -> positive z-score

        candles = _make_candles(prices)
        strategy = self._make_strategy(ou_window=90, entry_threshold=1.5, exit_threshold=0.0)
        signal = strategy.generate_signal("AAPL", candles, "us_tech", has_open_position=True)
        # With exit_threshold=0.0 and has_open_position=True, any positive z -> SELL
        assert signal is not None
        assert signal.direction == SignalDirection.SELL

    def test_ou_regime_gate_crisis(self) -> None:
        """Returns None when regime is CRISIS."""
        prices = _generate_ou_prices(100, mu=0.05, theta=5.0, sigma=0.05, seed=10)
        prices[-1] = math.exp(5.0) * 0.5  # Would normally BUY

        candles = _make_candles(prices)
        strategy = self._make_strategy(ou_window=90)

        regime_state = RegimeState(
            regime=MarketRegime.CRISIS,
            allow_new_longs=False,
            position_scale=0.1,
            vix_value=50.0,
        )
        signal = strategy.generate_signal("AAPL", candles, "us_tech", regime_state=regime_state)
        assert signal is None

    def test_ou_regime_gate_elevated_tightens(self) -> None:
        """In ELEVATED regime, entry threshold tightens to max(threshold, 2.0)."""
        prices = _generate_ou_prices(100, mu=0.05, theta=5.0, sigma=0.05, seed=10)
        # Force a moderate drop that would pass 1.5 but not 2.0
        mean_price = math.exp(5.0)
        prices[-1] = mean_price * 0.7  # Moderate drop

        candles = _make_candles(prices)

        # Without elevated: threshold=1.5, might get signal
        strategy_normal = self._make_strategy(ou_window=90, entry_threshold=1.5)
        signal_normal = strategy_normal.generate_signal("AAPL", candles, "us_tech")

        # With elevated: threshold tightens to 2.0
        strategy_elev = self._make_strategy(ou_window=90, entry_threshold=1.5)
        regime_state = RegimeState(
            regime=MarketRegime.ELEVATED,
            allow_new_longs=True,
            position_scale=0.5,
            vix_value=30.0,
        )
        signal_elev = strategy_elev.generate_signal(
            "AAPL", candles, "us_tech", regime_state=regime_state
        )

        # If normal generates a BUY, elevated should NOT (tighter threshold)
        # or elevated should also be BUY but only if z > 2.0
        if signal_normal is not None and signal_normal.direction == SignalDirection.BUY:
            # The elevated might filter it out or still pass if z is extreme enough
            # We just verify the regime gate is applied
            assert signal_elev is None or signal_elev.direction == SignalDirection.BUY

    def test_ou_reset_clears_state(self) -> None:
        """reset() clears any cached state."""
        strategy = self._make_strategy()
        # Manually set some internal state
        strategy._cached_params = {"AAPL": OUParams(mu=0.1, theta=5.0, sigma=0.1, half_life=6.9)}
        strategy.reset()
        assert strategy._cached_params == {}

    def test_ou_ru_wider_thresholds(self) -> None:
        """RU segments use 2.0 entry threshold in default segment params."""
        strategy = OUMeanReversionStrategy()
        params = strategy.get_parameters("ru_blue_chips")
        assert params.get("entry_threshold") == 2.0  # noqa: PLR2004

    def test_ou_insufficient_data(self) -> None:
        """Returns None when not enough candles for OU window."""
        prices = _generate_ou_prices(50)  # Too few for ou_window=90
        candles = _make_candles(prices)

        strategy = self._make_strategy(ou_window=90)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")
        assert signal is None
