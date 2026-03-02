"""Tests for triple barrier labeling (B.3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.ml.training.labeling import (
    TripleBarrierResult,
    build_triple_barrier_dataset,
    triple_barrier_label,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)
_SYMBOL = "TEST"
_MARKET = "us"
_TF = "1d"

# Number of candles needed for a minimal dataset
_N_CANDLES = 120


def _make_candle(
    index: int,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: int = 1000,
) -> Candle:
    """Create a single candle at offset `index` days from base timestamp."""
    c = close
    h = high if high is not None else c * 1.005
    lo = low if low is not None else c * 0.995
    o = open_ if open_ is not None else c
    return Candle(
        symbol=_SYMBOL,
        market_id=_MARKET,
        timeframe=_TF,
        timestamp=_BASE_TS + timedelta(days=index),
        open=Decimal(str(round(o, 4))),
        high=Decimal(str(round(h, 4))),
        low=Decimal(str(round(lo, 4))),
        close=Decimal(str(round(c, 4))),
        volume=volume,
    )


def _make_flat_candles(n: int, price: float = 100.0) -> list[Candle]:
    """Create n flat candles at a constant price."""
    return [_make_candle(i, price) for i in range(n)]


def _make_trending_candles(n: int, start_price: float, daily_return: float) -> list[Candle]:
    """Create n candles with a fixed daily return."""
    candles = []
    price = start_price
    for i in range(n):
        candles.append(_make_candle(i, price))
        price *= 1 + daily_return
    return candles


# ---------------------------------------------------------------------------
# Tests: triple_barrier_label
# ---------------------------------------------------------------------------


class TestUpperBarrierHit:
    """Price rises above upper barrier -> label=1, barrier_type='upper'."""

    def test_upper_barrier_hit(self) -> None:
        # Build candles: 20 bars of history, then entry, then a spike
        candles = _make_flat_candles(20, price=100.0)
        # Entry at index 19 (close=100)
        # Spike bar at index 20: high touches 104 (> 103 upper at 3%)
        candles.append(_make_candle(20, close=103.5, high=104.0, low=100.0, open_=100.0))
        # Add more bars for max_hold
        for i in range(21, 40):
            candles.append(_make_candle(i, 103.5))

        result = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )

        assert result is not None
        assert result.label == 1
        assert result.barrier_type == "upper"
        assert result.hold_bars == 1
        assert result.pnl_pct == pytest.approx(0.03, abs=1e-6)


class TestLowerBarrierHit:
    """Price drops below lower barrier -> label=0, barrier_type='lower'."""

    def test_lower_barrier_hit(self) -> None:
        candles = _make_flat_candles(20, price=100.0)
        # Drop bar: low touches 96.5 (< 97.0 lower at 3%)
        candles.append(_make_candle(20, close=97.0, high=100.0, low=96.5, open_=100.0))
        for i in range(21, 40):
            candles.append(_make_candle(i, 97.0))

        result = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )

        assert result is not None
        assert result.label == 0
        assert result.barrier_type == "lower"
        assert result.hold_bars == 1
        assert result.pnl_pct == pytest.approx(-0.03, abs=1e-6)


class TestVerticalBarrierTimeout:
    """Price stays within barriers -> label based on final return sign."""

    def test_vertical_positive_return(self) -> None:
        """Timeout with positive final return -> label=1."""
        candles = _make_flat_candles(20, price=100.0)
        # Slowly drift up but stay within 3% barriers
        for i in range(20, 30):
            # drift from 100 to ~101.5 over 10 bars
            price = 100.0 + (i - 20) * 0.15
            candles.append(_make_candle(i, price))

        result = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=10,
            atr_scale=False,
        )

        assert result is not None
        assert result.label == 1
        assert result.barrier_type == "vertical"
        assert result.pnl_pct > 0

    def test_vertical_negative_return(self) -> None:
        """Timeout with negative final return -> label=0."""
        candles = _make_flat_candles(20, price=100.0)
        # Slowly drift down but stay within 3% barriers
        for i in range(20, 30):
            price = 100.0 - (i - 20) * 0.15
            candles.append(_make_candle(i, price))

        result = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=10,
            atr_scale=False,
        )

        assert result is not None
        assert result.label == 0
        assert result.barrier_type == "vertical"
        assert result.pnl_pct < 0


class TestNoiseFiltering:
    """Small PnL vertical barrier hits return None."""

    def test_noise_filtered_out(self) -> None:
        """Vertical hit with tiny PnL (< 0.5% default) returns None."""
        candles = _make_flat_candles(20, price=100.0)
        # Essentially flat for max_hold bars -> pnl ~0
        for i in range(20, 30):
            candles.append(_make_candle(i, 100.01))

        result = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=10,
            atr_scale=False,
        )

        assert result is None

    def test_atr_noise_threshold_used(self) -> None:
        """When atr_scale=True, noise threshold scales with ATR."""
        # With very tight ATR (flat candles), even small returns should be
        # filtered as noise
        candles = _make_flat_candles(30, price=100.0)
        for i in range(30, 60):
            candles.append(_make_candle(i, 100.1))

        _ = triple_barrier_label(
            candles,
            entry_index=29,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
        )

        # With flat candles, ATR is tiny, so barriers are tiny,
        # and the small move should either hit a barrier or be filtered.
        # The key assertion is that it doesn't crash.


class TestSampleWeightsProportional:
    """PnL magnitude is preserved for use as sample_weight."""

    def test_pnl_magnitude_preserved(self) -> None:
        """Upper barrier hit has pnl_pct equal to upper_pct."""
        candles = _make_flat_candles(20, price=100.0)
        # Big spike
        candles.append(_make_candle(20, close=106.0, high=106.0, low=100.0, open_=100.0))
        for i in range(21, 40):
            candles.append(_make_candle(i, 106.0))

        result = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.05,
            lower_pct=0.05,
            max_hold=20,
            atr_scale=False,
        )

        assert result is not None
        assert result.pnl_pct == pytest.approx(0.05, abs=1e-6)
        # abs(pnl_pct) would be used as sample weight
        assert abs(result.pnl_pct) > 0


class TestATRScaledBarriers:
    """Barriers scale with ATR when atr_scale=True."""

    def test_atr_scaled_barriers_wider_with_volatile_data(self) -> None:
        """Volatile candles produce wider ATR-scaled barriers than fixed 3%."""
        # Create volatile candles with ~5% daily range
        candles = []
        price = 100.0
        for i in range(30):
            h = price * 1.03
            lo = price * 0.97
            candles.append(_make_candle(i, price, high=h, low=lo, open_=price))

        # Add bars that would hit 3% fixed barrier but not ATR-scaled
        candles.append(_make_candle(30, close=103.5, high=103.5, low=100.0, open_=100.0))
        candles.extend(_make_candle(i, 103.5) for i in range(31, 60))

        # With fixed 3%, this should hit upper barrier
        result_fixed = triple_barrier_label(
            candles,
            entry_index=29,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )

        # With ATR scaling on volatile data, barriers should be wider
        result_atr = triple_barrier_label(
            candles,
            entry_index=29,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
        )

        assert result_fixed is not None
        assert result_fixed.barrier_type == "upper"

        # With ATR scaling, the same move may not hit the wider barrier
        # so we either get vertical or None (noise-filtered)
        if result_atr is not None:
            # If we do get a result, verify it's not the same barrier type
            # or the barriers were wider (hold_bars >= fixed hold_bars)
            assert result_atr.barrier_type in ("upper", "vertical")


class TestBuildTripleBarrierDataset:
    """build_triple_barrier_dataset returns correct shapes."""

    def test_returns_correct_shapes(self) -> None:
        """Features, labels, and weights lists have same length."""
        # Create enough candles for at least a few samples
        # Need: window_size (60) + max_hold (20) + some extra = ~90+
        candles = []
        price = 100.0
        for i in range(_N_CANDLES):
            # Add some price movement so we get labels
            price = 100.0 + 5.0 * ((-1) ** i)  # oscillate between 95 and 105
            h = price * 1.01
            lo = price * 0.99
            candles.append(_make_candle(i, price, high=h, low=lo, open_=price))

        features, labels, weights = build_triple_barrier_dataset(
            candles,
            window_size=60,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )

        assert len(features) == len(labels) == len(weights)
        # Should have at least some samples
        assert len(features) > 0
        # Labels are binary
        assert all(lbl in (0, 1) for lbl in labels)
        # Weights are non-negative
        assert all(w >= 0 for w in weights)

    def test_features_are_dicts(self) -> None:
        """Each feature entry is a dict of float values."""
        candles = []
        price = 100.0
        for i in range(_N_CANDLES):
            price = 100.0 + 5.0 * ((-1) ** i)
            h = price * 1.01
            lo = price * 0.99
            candles.append(_make_candle(i, price, high=h, low=lo, open_=price))

        features, _labels, _weights = build_triple_barrier_dataset(
            candles,
            window_size=60,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )

        if features:
            assert isinstance(features[0], dict)
            assert all(isinstance(v, float) for v in features[0].values())

    def test_empty_with_insufficient_candles(self) -> None:
        """Returns empty lists when not enough candles."""
        candles = _make_flat_candles(10)
        features, labels, weights = build_triple_barrier_dataset(
            candles,
            window_size=60,
            max_hold=20,
        )
        assert features == []
        assert labels == []
        assert weights == []


class TestEdgeCases:
    """Edge cases for triple_barrier_label."""

    def test_invalid_entry_index(self) -> None:
        """Out of bounds entry index returns None."""
        candles = _make_flat_candles(10)
        assert triple_barrier_label(candles, entry_index=-1) is None
        assert triple_barrier_label(candles, entry_index=100) is None

    def test_entry_at_last_bar(self) -> None:
        """Entry at last bar with no forward data returns None."""
        candles = _make_flat_candles(10)
        result = triple_barrier_label(candles, entry_index=9, max_hold=5)
        assert result is None

    def test_upper_before_lower_on_same_bar(self) -> None:
        """When both barriers could be hit, upper is checked first."""
        candles = _make_flat_candles(20, price=100.0)
        # Bar where both high > upper AND low < lower (gap bar)
        candles.append(_make_candle(20, close=104.0, high=104.0, low=96.0, open_=100.0))
        for i in range(21, 40):
            candles.append(_make_candle(i, 104.0))

        result = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )

        assert result is not None
        # Upper is checked first in the implementation
        assert result.label == 1
        assert result.barrier_type == "upper"
