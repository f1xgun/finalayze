"""Tests for ruble-oil decorrelation regime signal."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle


def _make_candle(
    symbol: str,
    close: Decimal,
    idx: int,
    *,
    market_id: str = "moex",
    timeframe: str = "1d",
) -> Candle:
    """Build a synthetic candle with the given close price."""
    ts = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=idx)
    return Candle(
        symbol=symbol,
        market_id=market_id,
        timeframe=timeframe,
        timestamp=ts,
        open=close,
        high=close + Decimal("0.5"),
        low=close - Decimal("0.5"),
        close=close,
        volume=1000,
    )


def _make_correlated_candles(
    n: int,
    correlation: float,
) -> tuple[list[Candle], list[Candle]]:
    """Generate two series of candles with approximately the given correlation.

    Uses a common factor model: rub = a*common + (1-a)*noise_rub,
    oil = a*common + (1-a)*noise_oil, where a controls correlation.
    """
    import random

    rng = random.Random(42)  # noqa: S311

    # Determine mixing coefficient from desired correlation
    # correlation ~ a^2 / (a^2 + (1-a)^2) ... approximate, but we use a simpler model:
    # X = a*Z + sqrt(1-a^2)*E1, Y = a*Z + sqrt(1-a^2)*E2 => corr(X,Y) = a^2
    a = math.sqrt(abs(correlation))
    if correlation < 0:
        a = -a
    b = math.sqrt(1 - a * a) if abs(a) < 1 else 0.0

    base_rub = 75.0
    base_oil = 80.0

    rub_candles: list[Candle] = []
    oil_candles: list[Candle] = []

    for i in range(n):
        common = rng.gauss(0, 0.01)
        noise_rub = rng.gauss(0, 0.01)
        noise_oil = rng.gauss(0, 0.01)

        ret_rub = a * common + b * noise_rub
        ret_oil = a * common + b * noise_oil

        base_rub *= 1 + ret_rub
        base_oil *= 1 + ret_oil

        rub_candles.append(_make_candle("USDRUB", Decimal(str(round(base_rub, 4))), i))
        oil_candles.append(
            _make_candle("BRN", Decimal(str(round(base_oil, 4))), i, market_id="ice")
        )

    return rub_candles, oil_candles


class TestComputeRubOilCorrelation:
    """Tests for the compute_rub_oil_correlation helper function."""

    def test_compute_correlation_returns_float(self) -> None:
        """Helper function returns a valid float in [-1, 1]."""
        from finalayze.risk.rub_oil_regime import compute_rub_oil_correlation

        rub, oil = _make_correlated_candles(70, 0.8)
        result = compute_rub_oil_correlation(rub, oil, window=60)

        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_one(self) -> None:
        """With fewer than window+1 candles, returns 1.0 (assume normal)."""
        from finalayze.risk.rub_oil_regime import compute_rub_oil_correlation

        rub, oil = _make_correlated_candles(30, 0.5)
        result = compute_rub_oil_correlation(rub, oil, window=60)

        assert result == 1.0

    def test_high_correlation_value(self) -> None:
        """Highly correlated series should produce correlation > 0.5."""
        from finalayze.risk.rub_oil_regime import compute_rub_oil_correlation

        rub, oil = _make_correlated_candles(100, 0.9)
        result = compute_rub_oil_correlation(rub, oil, window=60)

        assert result > 0.5


class TestRubOilRegimeSignal:
    """Tests for the RubOilRegimeSignal class."""

    def test_high_correlation_returns_normal(self) -> None:
        """Correlation > 0.3 should return NORMAL regime."""
        from finalayze.risk.regime import MarketRegime
        from finalayze.risk.rub_oil_regime import RubOilRegimeSignal

        rub, oil = _make_correlated_candles(100, 0.9)
        provider = RubOilRegimeSignal(rub_candles=rub, oil_candles=oil)

        # Asset candles (dummy -- provider uses its own rub/oil data)
        asset_candles = [_make_candle("GAZP", Decimal(150), i) for i in range(100)]
        state = provider.get_regime(asset_candles, bar_index=99)

        assert state.regime == MarketRegime.NORMAL
        assert state.allow_new_longs is True
        assert state.position_scale == Decimal("1.0")

    def test_low_correlation_returns_elevated(self) -> None:
        """Correlation 0.1-0.3 should return ELEVATED regime."""
        from finalayze.risk.regime import MarketRegime
        from finalayze.risk.rub_oil_regime import RubOilRegimeSignal

        # correlation ~ 0.04 (a^2 where a=0.2), well below 0.3 but we need 0.1-0.3
        # Use correlation=0.2 so corr ~ 0.2
        rub, oil = _make_correlated_candles(100, 0.2)
        provider = RubOilRegimeSignal(rub_candles=rub, oil_candles=oil)

        asset_candles = [_make_candle("GAZP", Decimal(150), i) for i in range(100)]
        state = provider.get_regime(asset_candles, bar_index=99)

        assert state.regime == MarketRegime.ELEVATED
        assert state.allow_new_longs is True
        assert state.position_scale == Decimal("0.5")

    def test_very_low_correlation_returns_crisis(self) -> None:
        """Correlation < 0.1 should return CRISIS regime."""
        from finalayze.risk.regime import MarketRegime
        from finalayze.risk.rub_oil_regime import RubOilRegimeSignal

        # Near-zero correlation
        rub, oil = _make_correlated_candles(100, 0.01)
        provider = RubOilRegimeSignal(rub_candles=rub, oil_candles=oil)

        asset_candles = [_make_candle("GAZP", Decimal(150), i) for i in range(100)]
        state = provider.get_regime(asset_candles, bar_index=99)

        assert state.regime == MarketRegime.CRISIS
        assert state.allow_new_longs is False
        assert state.position_scale == Decimal("0.25")

    def test_insufficient_data_returns_normal(self) -> None:
        """Fewer than 61 candles should return NORMAL regime."""
        from finalayze.risk.regime import MarketRegime
        from finalayze.risk.rub_oil_regime import RubOilRegimeSignal

        rub, oil = _make_correlated_candles(30, 0.01)
        provider = RubOilRegimeSignal(rub_candles=rub, oil_candles=oil)

        asset_candles = [_make_candle("GAZP", Decimal(150), i) for i in range(30)]
        state = provider.get_regime(asset_candles, bar_index=29)

        assert state.regime == MarketRegime.NORMAL
        assert state.allow_new_longs is True

    def test_protocol_conformance(self) -> None:
        """RubOilRegimeSignal must satisfy the RegimeProvider protocol."""
        from finalayze.risk.regime import RegimeProvider
        from finalayze.risk.rub_oil_regime import RubOilRegimeSignal

        rub, oil = _make_correlated_candles(10, 0.5)
        provider = RubOilRegimeSignal(rub_candles=rub, oil_candles=oil)

        # Protocol check: must have get_regime(candles, bar_index) -> RegimeState
        assert hasattr(provider, "get_regime")
        assert callable(provider.get_regime)

        # Structural subtyping check
        def accepts_provider(p: RegimeProvider) -> None:
            pass

        # This should not raise TypeError at runtime
        accepts_provider(provider)
