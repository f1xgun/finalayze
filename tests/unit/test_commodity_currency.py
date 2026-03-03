"""Tests for commodity-currency premium signal."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.risk.commodity_currency import (
    compute_commodity_currency_premium,
    compute_premium_confidence_boost,
    is_energy_undervalued,
)

_WINDOW = 20


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


def _make_series(
    symbol: str,
    start_price: float,
    end_price: float,
    n: int,
    *,
    market_id: str = "moex",
) -> list[Candle]:
    """Generate a linear series of candles from start_price to end_price with n candles."""
    candles: list[Candle] = []
    for i in range(n):
        price = start_price + (end_price - start_price) * i / (n - 1) if n > 1 else start_price
        candles.append(
            _make_candle(
                symbol,
                Decimal(str(round(price, 4))),
                i,
                market_id=market_id,
            )
        )
    return candles


class TestComputeCommodityCurrencyPremium:
    """Tests for compute_commodity_currency_premium."""

    def test_positive_premium_when_oil_up_rub_stable(self) -> None:
        """Oil goes up 10%, RUB stable -> positive premium ~0.10."""
        n = _WINDOW + 1
        oil_candles = _make_series("BRENT", 80.0, 88.0, n)  # +10%
        rub_candles = _make_series("USDRUB", 90.0, 90.0, n)  # 0%

        premium = compute_commodity_currency_premium(oil_candles, rub_candles, window=_WINDOW)

        assert premium == pytest.approx(0.10, abs=0.001)

    def test_negative_premium_when_rub_weakens_more(self) -> None:
        """RUB weakens 15%, oil gains 5% -> premium = 0.05 - 0.15 = -0.10."""
        n = _WINDOW + 1
        oil_candles = _make_series("BRENT", 80.0, 84.0, n)  # +5%
        rub_candles = _make_series("USDRUB", 80.0, 92.0, n)  # +15% (RUB weakens)

        premium = compute_commodity_currency_premium(oil_candles, rub_candles, window=_WINDOW)

        assert premium == pytest.approx(-0.10, abs=0.001)

    def test_insufficient_data_returns_zero(self) -> None:
        """Too few candles for the window -> returns 0.0."""
        # Need window+1 candles, provide only window
        oil_candles = _make_series("BRENT", 80.0, 88.0, _WINDOW)
        rub_candles = _make_series("USDRUB", 90.0, 90.0, _WINDOW)

        premium = compute_commodity_currency_premium(oil_candles, rub_candles, window=_WINDOW)

        assert premium == 0.0

    def test_zero_start_price_returns_zero(self) -> None:
        """Zero start price should not cause division error."""
        n = _WINDOW + 1
        oil_candles = _make_series("BRENT", 0.0, 10.0, n)
        rub_candles = _make_series("USDRUB", 90.0, 90.0, n)
        # oil_start is 0.0 -> guard returns 0.0
        premium = compute_commodity_currency_premium(oil_candles, rub_candles, window=_WINDOW)

        assert premium == 0.0


class TestIsEnergyUndervalued:
    """Tests for is_energy_undervalued."""

    def test_true_above_threshold(self) -> None:
        """Premium > 5% threshold -> True."""
        n = _WINDOW + 1
        oil_candles = _make_series("BRENT", 80.0, 88.0, n)  # +10%
        rub_candles = _make_series("USDRUB", 90.0, 90.0, n)  # 0%

        assert is_energy_undervalued(oil_candles, rub_candles, window=_WINDOW) is True

    def test_false_below_threshold(self) -> None:
        """Premium < 5% threshold -> False."""
        n = _WINDOW + 1
        oil_candles = _make_series("BRENT", 80.0, 82.0, n)  # +2.5%
        rub_candles = _make_series("USDRUB", 90.0, 90.0, n)  # 0%

        assert is_energy_undervalued(oil_candles, rub_candles, window=_WINDOW) is False

    def test_false_at_exact_threshold(self) -> None:
        """Premium == threshold -> False (strictly greater than required)."""
        n = _WINDOW + 1
        # Exactly 5%
        oil_candles = _make_series("BRENT", 100.0, 105.0, n)
        rub_candles = _make_series("USDRUB", 90.0, 90.0, n)

        assert is_energy_undervalued(oil_candles, rub_candles, window=_WINDOW) is False


class TestComputePremiumConfidenceBoost:
    """Tests for compute_premium_confidence_boost."""

    def test_boost_above_threshold(self) -> None:
        """Premium exceeds threshold -> returns positive boost."""
        n = _WINDOW + 1
        oil_candles = _make_series("BRENT", 80.0, 88.0, n)  # +10%
        rub_candles = _make_series("USDRUB", 90.0, 90.0, n)  # 0%

        boost = compute_premium_confidence_boost(oil_candles, rub_candles, window=_WINDOW)

        assert boost > 0.0

    def test_boost_zero_below_threshold(self) -> None:
        """Premium below threshold -> returns 0.0."""
        n = _WINDOW + 1
        oil_candles = _make_series("BRENT", 80.0, 82.0, n)  # +2.5%
        rub_candles = _make_series("USDRUB", 90.0, 90.0, n)  # 0%

        boost = compute_premium_confidence_boost(oil_candles, rub_candles, window=_WINDOW)

        assert boost == 0.0

    def test_boost_capped_at_max(self) -> None:
        """Boost never exceeds max_boost even with huge premium."""
        n = _WINDOW + 1
        max_boost = 0.10
        oil_candles = _make_series("BRENT", 80.0, 160.0, n)  # +100%
        rub_candles = _make_series("USDRUB", 90.0, 90.0, n)  # 0%

        boost = compute_premium_confidence_boost(
            oil_candles, rub_candles, window=_WINDOW, max_boost=max_boost
        )

        assert boost == pytest.approx(max_boost, abs=1e-9)
