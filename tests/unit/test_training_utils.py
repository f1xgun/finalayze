"""Tests for shared ML training utilities."""

from __future__ import annotations

import datetime
from decimal import Decimal

from finalayze.core.schemas import Candle
from finalayze.ml.training import DEFAULT_WINDOW_SIZE, build_windows


def _make_candles(n: int, base_price: float = 100.0) -> list[Candle]:
    """Create n synthetic candles with small random-ish increments."""
    candles: list[Candle] = []
    for i in range(n):
        price = Decimal(str(base_price + i * 0.5))
        candles.append(
            Candle(
                symbol="TEST",
                market_id="us",
                timeframe="1d",
                timestamp=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)
                + datetime.timedelta(days=i),
                open=price - Decimal("0.1"),
                high=price + Decimal(1),
                low=price - Decimal(1),
                close=price,
                volume=1000,
            )
        )
    return candles


class TestBuildWindows:
    def test_label_outside_feature_window(self) -> None:
        """Label is derived from bar *after* the feature window (no overlap)."""
        candles = _make_candles(DEFAULT_WINDOW_SIZE + 2)
        features, labels, timestamps = build_windows(candles, DEFAULT_WINDOW_SIZE)

        # With WINDOW_SIZE + 2 candles, we get 2 samples
        assert len(features) == 2
        assert len(labels) == 2
        assert len(timestamps) == 2

        # Label for first sample: sign(candles[60].close - candles[59].close)
        # Since price increases monotonically, label should be 1 (BUY)
        assert labels[0] == 1

    def test_insufficient_candles_returns_empty(self) -> None:
        """When fewer than window_size + 1 candles, return empty lists."""
        candles = _make_candles(DEFAULT_WINDOW_SIZE)  # exactly window_size, need +1
        features, labels, timestamps = build_windows(candles, DEFAULT_WINDOW_SIZE)
        assert features == []
        assert labels == []
        assert timestamps == []

    def test_empty_candles(self) -> None:
        """Empty candle list should return empty results."""
        features, labels, timestamps = build_windows([], DEFAULT_WINDOW_SIZE)
        assert features == []
        assert labels == []
        assert timestamps == []

    def test_custom_window_size(self) -> None:
        """build_windows should respect custom window_size parameter."""
        small_window = 30
        candles = _make_candles(small_window + 5)
        features, labels, timestamps = build_windows(candles, small_window)
        assert len(features) == 5
        assert len(labels) == 5
        assert len(timestamps) == 5
