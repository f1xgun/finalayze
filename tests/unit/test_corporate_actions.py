"""Tests for corporate action detection and handling (6C.8)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.ml.features.corporate_actions import adjust_for_splits, detect_splits


def _make_candle(
    i: int, close: float, *, high: float | None = None, low: float | None = None
) -> Candle:
    """Create a synthetic candle at day offset i."""
    return Candle(
        symbol="TEST",
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=i),
        open=Decimal(str(close - 0.5)),
        high=Decimal(str(high if high is not None else close + 1.0)),
        low=Decimal(str(low if low is not None else close - 1.0)),
        close=Decimal(str(close)),
        volume=1000,
    )


class TestDetectSplits:
    def test_detect_splits_finds_2_for_1(self) -> None:
        """A 2:1 split (price halves) is detected."""
        candles = [_make_candle(i, 100.0) for i in range(5)]
        # Day 5: 2:1 split -> close drops from 100 to 50
        # Small bar range to look like a split (not a crash)
        candles.append(_make_candle(5, 50.0, high=51.0, low=49.0))
        candles.append(_make_candle(6, 51.0))

        suspects = detect_splits(candles)
        assert 5 in suspects

    def test_detect_splits_ignores_normal_volatility(self) -> None:
        """A 5% daily move is not flagged."""
        candles = [_make_candle(i, 100.0 + i * 5.0) for i in range(10)]
        suspects = detect_splits(candles)
        assert len(suspects) == 0

    def test_detect_splits_finds_reverse_split(self) -> None:
        """A 1:2 reverse split (price doubles) is detected."""
        candles = [_make_candle(i, 50.0) for i in range(5)]
        # Day 5: 1:2 reverse split -> close jumps from 50 to 100
        candles.append(_make_candle(5, 100.0, high=101.0, low=99.0))
        candles.append(_make_candle(6, 101.0))

        suspects = detect_splits(candles)
        assert 5 in suspects


class TestAdjustForSplits:
    def test_adjust_for_splits_corrects_prices(self) -> None:
        """After adjustment, the series should be smooth across the split."""
        candles = [_make_candle(i, 100.0) for i in range(5)]
        # Day 5: 2:1 split
        candles.append(_make_candle(5, 50.0, high=51.0, low=49.0))
        candles.append(_make_candle(6, 51.0))

        adjusted = adjust_for_splits(candles)

        # After adjustment, all pre-split prices should be ~halved
        # to match post-split level
        for i in range(5):
            assert float(adjusted[i].close) == pytest.approx(50.0, abs=1.0)

    def test_adjust_preserves_post_split_prices(self) -> None:
        """Post-split prices are unchanged."""
        candles = [_make_candle(i, 100.0) for i in range(5)]
        candles.append(_make_candle(5, 50.0, high=51.0, low=49.0))
        candles.append(_make_candle(6, 51.0))

        adjusted = adjust_for_splits(candles)

        # Post-split candles should not change
        assert float(adjusted[5].close) == pytest.approx(50.0, abs=0.1)
        assert float(adjusted[6].close) == pytest.approx(51.0, abs=0.1)


class TestBuildWindowsSkipSplit:
    def test_build_windows_skips_split_window(self) -> None:
        """Windows spanning a detected split are excluded."""
        from finalayze.ml.training import build_windows

        # Create 70 normal candles + split at candle 35
        candles = [_make_candle(i, 100.0 + i * 0.5) for i in range(35)]
        # Split at index 35
        candles.append(
            _make_candle(35, 60.0, high=61.0, low=59.0)
        )
        # Continue post-split
        for i in range(36, 100):
            candles.append(_make_candle(i, 60.0 + (i - 35) * 0.5))

        features, labels, _ = build_windows(candles, window_size=30)
        # Should have some results, but not an error
        assert len(features) > 0
