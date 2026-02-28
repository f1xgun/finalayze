"""Tests for multi-symbol temporal ordering in build_dataset (5.7)."""

from __future__ import annotations

import datetime
from decimal import Decimal
from unittest.mock import patch

from finalayze.core.schemas import Candle
from finalayze.ml.training import DEFAULT_WINDOW_SIZE, build_dataset, build_windows

# Use a window large enough for compute_features (needs ~30+ bars for indicators)
_W = DEFAULT_WINDOW_SIZE


def _make_candles(
    n: int,
    symbol: str = "TEST",
    base_price: float = 100.0,
    start_date: datetime.datetime | None = None,
) -> list[Candle]:
    """Create n synthetic candles."""
    if start_date is None:
        start_date = datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)
    candles: list[Candle] = []
    for i in range(n):
        price = Decimal(str(base_price + i * 0.5))
        candles.append(
            Candle(
                symbol=symbol,
                market_id="us",
                timeframe="1d",
                timestamp=start_date + datetime.timedelta(days=i),
                open=price - Decimal("0.1"),
                high=price + Decimal(1),
                low=price - Decimal(1),
                close=price,
                volume=1000,
            )
        )
    return candles


def _mock_features(candles: list[Candle], sentiment_score: float = 0.0) -> dict[str, float]:
    """Fake compute_features that always works."""
    return {"close": float(candles[-1].close), "vol": float(candles[-1].volume)}


class TestBuildWindowsTimestamps:
    def test_returns_timestamps(self) -> None:
        """build_windows should return 3-tuple with timestamps."""
        candles = _make_candles(_W + 3)
        with patch("finalayze.ml.training.compute_features", side_effect=_mock_features):
            features, labels, timestamps = build_windows(candles, _W)

        assert len(timestamps) == len(features) == len(labels)
        assert len(timestamps) == 3

    def test_timestamps_match_label_bars(self) -> None:
        """Timestamps should correspond to the label bar (candles[i+window_size])."""
        candles = _make_candles(_W + 2)
        with patch("finalayze.ml.training.compute_features", side_effect=_mock_features):
            _features, _labels, timestamps = build_windows(candles, _W)

        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        assert timestamps[0] == sorted_candles[_W].timestamp
        assert timestamps[1] == sorted_candles[_W + 1].timestamp


class TestBuildDatasetTemporalOrdering:
    def test_multi_symbol_sorted_by_timestamp(self) -> None:
        """Features from multiple symbols should be interleaved by timestamp."""
        # Symbol A starts Jan 1
        candles_a = _make_candles(
            _W + 3,
            symbol="A",
            start_date=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
        )
        # Symbol B starts Jan 2 (one day later)
        candles_b = _make_candles(
            _W + 3,
            symbol="B",
            base_price=200.0,
            start_date=datetime.datetime(2025, 1, 2, tzinfo=datetime.UTC),
        )

        with patch("finalayze.ml.training.compute_features", side_effect=_mock_features):
            _features, _labels, timestamps = build_dataset({"A": candles_a, "B": candles_b}, _W)

        # Timestamps should be monotonically non-decreasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Temporal order violated at index {i}: {timestamps[i - 1]} > {timestamps[i]}"
            )

    def test_single_symbol_same_as_build_windows(self) -> None:
        """With one symbol, build_dataset should match build_windows."""
        candles = _make_candles(_W + 4, symbol="X")

        with patch("finalayze.ml.training.compute_features", side_effect=_mock_features):
            f1, l1, t1 = build_windows(candles, _W)
            f2, l2, t2 = build_dataset({"X": candles}, _W)

        assert f1 == f2
        assert l1 == l2
        assert t1 == t2
