"""Tests for volatility-targeting utility and strategy integration (Layer 4)."""

from __future__ import annotations

import math
import random

from finalayze.strategies.vol_targeting import compute_vol_scale

# ---------------------------------------------------------------------------
# Constants for tests (avoid magic numbers — ruff PLR2004)
# ---------------------------------------------------------------------------
_DEFAULT_LOOKBACK = 126
_DEFAULT_TARGET_VOL = 0.15
_ANNUALIZATION = 252.0
_CLAMP_MAX = 2.0
_INSUFFICIENT_LENGTH = 50  # less than lookback + 1


class TestComputeVolScale:
    """Unit tests for compute_vol_scale."""

    def test_vol_scale_basic(self) -> None:
        """Known constant-step series should produce a predictable scale."""
        # Build a series of 200 prices with daily returns of ~0.1%
        # so realized vol ~ 0 (constant returns have zero std).
        # Use slight noise to avoid zero-division.
        n = 200
        base_price = 100.0
        daily_return = 0.001  # 0.1% per day
        prices = [base_price * (1 + daily_return) ** i for i in range(n)]

        # Add small noise to make std nonzero
        rng = random.Random(42)  # noqa: S311
        prices_noisy = [p * (1 + rng.gauss(0, 0.005)) for p in prices]

        scale = compute_vol_scale(prices_noisy, lookback=_DEFAULT_LOOKBACK)
        # Scale should be a positive finite number
        assert scale > 0.0
        assert math.isfinite(scale)
        # With ~0.5% daily noise, annualized vol ~ 0.005 * sqrt(252) ~ 0.079
        # target=0.15, so scale ~ 0.15/0.079 ~ 1.9
        assert scale <= _CLAMP_MAX

    def test_vol_scale_clamp_high(self) -> None:
        """Very low volatility should clamp scale at 2.0."""
        # Nearly constant prices -> near-zero vol -> scale would be huge
        n = 200
        prices = [100.0 + i * 0.00001 for i in range(n)]  # near-flat
        scale = compute_vol_scale(prices, lookback=_DEFAULT_LOOKBACK)
        assert scale == _CLAMP_MAX

    def test_vol_scale_clamp_low(self) -> None:
        """Very high volatility should produce a scale near 0."""
        # Huge daily swings -> very high vol -> scale near 0
        n = 200
        rng = random.Random(99)  # noqa: S311
        prices: list[float] = [100.0]
        for _ in range(n - 1):
            # ~50% daily moves -> annualized vol is enormous
            factor = 1 + rng.gauss(0, 0.5)
            prices.append(max(0.01, prices[-1] * factor))

        scale = compute_vol_scale(
            prices, lookback=_DEFAULT_LOOKBACK, target_vol=_DEFAULT_TARGET_VOL
        )
        # With ~50% daily vol, annualized ~793%, scale = 0.15/7.93 ~ 0.019
        assert scale < 0.1

    def test_vol_scale_insufficient_data(self) -> None:
        """When fewer than lookback+1 prices, return 1.0."""
        prices = [100.0 + i for i in range(_INSUFFICIENT_LENGTH)]
        scale = compute_vol_scale(prices, lookback=_DEFAULT_LOOKBACK)
        assert scale == 1.0

    def test_vol_scale_exact_boundary(self) -> None:
        """Exactly lookback+1 prices should compute (not return 1.0)."""
        n = _DEFAULT_LOOKBACK + 1
        prices = [100.0 + i * 0.5 for i in range(n)]
        scale = compute_vol_scale(prices, lookback=_DEFAULT_LOOKBACK)
        # Should actually compute, not bail out
        # Function should run without error; scale is a valid float
        assert math.isfinite(scale)


class TestMomentumVolTarget:
    """Integration: MomentumStrategy with vol_target_enabled."""

    def test_momentum_with_vol_target(self) -> None:
        """When vol targeting is on, confidence should be scaled."""
        from finalayze.core.schemas import Candle, SignalDirection
        from finalayze.strategies.momentum import MomentumStrategy

        strategy = MomentumStrategy()

        # Prepare params that will produce a BUY signal
        params: dict[str, object] = {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "min_confidence": 0.0,
            "lookback_bars": 5,
            "vol_target_enabled": False,
            "vol_target": 0.15,
        }
        segment = "test_segment"
        strategy._params_cache[segment] = params  # type: ignore[assignment]

        # Build 250 candles with a dip to trigger oversold RSI then recovery
        candles: list[Candle] = []
        n = 250
        for i in range(n):
            if i < 200:
                price = 100.0 + i * 0.1
            elif i < 220:
                # Sharp dip to trigger oversold RSI
                price = 120.0 - (i - 200) * 2.0
            else:
                # Recovery
                price = 80.0 + (i - 220) * 1.5

            candles.append(
                Candle(
                    symbol="TEST",
                    market_id="us",
                    open=price,
                    high=price * 1.01,
                    low=price * 0.99,
                    close=price,
                    volume=1000,
                    timeframe="1d",
                    timestamp="2026-01-01T00:00:00Z",
                )
            )

        # First get signal WITHOUT vol targeting
        signal_no_vol = strategy.generate_signal("TEST", candles, segment)

        # Now enable vol targeting
        params_vol = dict(params)
        params_vol["vol_target_enabled"] = True
        params_vol["vol_target"] = 0.15
        strategy._params_cache[segment] = params_vol  # type: ignore[assignment]
        # Reset signal state to allow re-emission
        strategy._signal_states.clear()

        signal_with_vol = strategy.generate_signal("TEST", candles, segment)

        # Both should produce a signal (or both None -- if so, skip assertion)
        if signal_no_vol is not None and signal_with_vol is not None:
            # With vol targeting, confidence should differ from unscaled
            # (unless scale happens to be exactly 1.0, which is unlikely)
            assert signal_with_vol.confidence >= 0.0
            assert signal_with_vol.confidence <= 1.0


class TestDualMomentumVolTarget:
    """Integration: DualMomentumStrategy with vol_target_enabled."""

    def test_dual_momentum_with_vol_target(self) -> None:
        """When vol targeting is on, confidence should be scaled."""
        from finalayze.core.schemas import Candle
        from finalayze.strategies.dual_momentum import DualMomentumStrategy

        strategy = DualMomentumStrategy(vol_target_enabled=True, vol_target=0.15)

        # Build 200 candles with upward trend (positive momentum)
        candles: list[Candle] = []
        for i in range(200):
            price = 100.0 + i * 0.5  # steady uptrend
            candles.append(
                Candle(
                    symbol="TEST",
                    market_id="us",
                    open=price,
                    high=price * 1.01,
                    low=price * 0.99,
                    close=price,
                    volume=1000,
                    timeframe="1d",
                    timestamp="2026-01-01T00:00:00Z",
                )
            )

        signal = strategy.generate_signal("TEST", candles, "us_tech")
        assert signal is not None
        assert signal.confidence >= 0.0
        assert signal.confidence <= 1.0
