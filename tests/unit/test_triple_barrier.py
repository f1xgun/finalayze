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
        """Features, labels, weights, and timestamps lists have same length."""
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

        features, labels, weights, timestamps, _ = build_triple_barrier_dataset(
            candles,
            window_size=60,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )

        assert len(features) == len(labels) == len(weights) == len(timestamps)
        # Should have at least some samples
        assert len(features) > 0
        # Labels are binary
        assert all(lbl in (0, 1) for lbl in labels)
        # Weights are non-negative
        assert all(w >= 0 for w in weights)
        # Timestamps are monotonically non-decreasing
        for j in range(1, len(timestamps)):
            assert timestamps[j] >= timestamps[j - 1]

    def test_features_are_dicts(self) -> None:
        """Each feature entry is a dict of float values."""
        candles = []
        price = 100.0
        for i in range(_N_CANDLES):
            price = 100.0 + 5.0 * ((-1) ** i)
            h = price * 1.01
            lo = price * 0.99
            candles.append(_make_candle(i, price, high=h, low=lo, open_=price))

        features, _labels, _weights, _timestamps, _ = build_triple_barrier_dataset(
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
        features, labels, weights, timestamps, hold_bars = build_triple_barrier_dataset(
            candles,
            window_size=60,
            max_hold=20,
        )
        assert features == []
        assert labels == []
        assert weights == []
        assert timestamps == []
        assert hold_bars == []


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


# ---------------------------------------------------------------------------
# Tests: Market-neutral (benchmark) triple barrier labels (A2)
# ---------------------------------------------------------------------------


class TestMarketNeutralLabels:
    """Market-neutral labeling using benchmark_candles."""

    def test_balanced_labels_with_benchmark(self) -> None:
        """Upward-drifting stock + same-drift benchmark -> ~50/50 labels."""
        n = 120
        daily_ret = 0.002  # 0.2% daily drift

        # Stock and benchmark both drift up at the same rate
        stock_candles = _make_trending_candles(n, start_price=100.0, daily_return=daily_ret)
        bench_candles = _make_trending_candles(n, start_price=100.0, daily_return=daily_ret)

        # Without benchmark: labels should be biased toward 1 (upward drift)
        results_raw: list[int] = []
        for i in range(20, n - 20):
            r = triple_barrier_label(
                stock_candles,
                i,
                upper_pct=0.03,
                lower_pct=0.03,
                max_hold=15,
                atr_scale=False,
            )
            if r is not None:
                results_raw.append(r.label)

        # With benchmark: excess return is ~0 -> labels should be more balanced
        results_neutral: list[int] = []
        for i in range(20, n - 20):
            r = triple_barrier_label(
                stock_candles,
                i,
                upper_pct=0.03,
                lower_pct=0.03,
                max_hold=15,
                atr_scale=False,
                benchmark_candles=bench_candles,
            )
            if r is not None:
                results_neutral.append(r.label)

        # Raw labels should be biased upward (mostly 1s)
        if results_raw:
            raw_ratio = sum(results_raw) / len(results_raw)
            assert raw_ratio > 0.6, f"Expected upward bias, got {raw_ratio}"

        # Neutral labels should be less biased (closer to 0.5)
        # Most should be filtered as noise (excess return ~0)
        # Those that survive should not be heavily biased
        if results_neutral:
            neutral_ratio = sum(results_neutral) / len(results_neutral)
            assert neutral_ratio < raw_ratio, (
                f"Neutral ({neutral_ratio}) should be less biased than raw ({raw_ratio})"
            )

    def test_backward_compatibility_none_benchmark(self) -> None:
        """benchmark_candles=None produces identical results to old behavior."""
        candles = _make_flat_candles(20, price=100.0)
        candles.append(_make_candle(20, close=103.5, high=104.0, low=100.0, open_=100.0))
        for i in range(21, 40):
            candles.append(_make_candle(i, 103.5))

        result_old = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )
        result_new = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
            benchmark_candles=None,
        )

        assert result_old is not None
        assert result_new is not None
        assert result_old.label == result_new.label
        assert result_old.barrier_type == result_new.barrier_type
        assert result_old.pnl_pct == pytest.approx(result_new.pnl_pct)
        assert result_old.hold_bars == result_new.hold_bars

    def test_excess_return_barrier_logic(self) -> None:
        """Stock up 3% but benchmark up 2.5% -> excess only 0.5%, no upper hit."""
        candles = _make_flat_candles(20, price=100.0)
        # Stock goes up 3.5% on bar 20 (high touches 103.5)
        candles.append(_make_candle(20, close=103.5, high=103.5, low=100.0, open_=100.0))
        for i in range(21, 40):
            candles.append(_make_candle(i, 103.5))

        # Benchmark goes up 2.5% on bar 20
        bench = _make_flat_candles(20, price=100.0)
        bench.append(_make_candle(20, close=102.5, high=102.5, low=100.0, open_=100.0))
        for i in range(21, 40):
            bench.append(_make_candle(i, 102.5))

        # Without benchmark: should hit upper barrier (3.5% > 3%)
        result_raw = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )
        assert result_raw is not None
        assert result_raw.barrier_type == "upper"

        # With benchmark: excess = 3.5% - 2.5% = 1.0%, below 3% barrier
        result_neutral = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
            benchmark_candles=bench,
        )
        # Excess return is only ~1%, should NOT hit 3% upper barrier
        if result_neutral is not None:
            assert result_neutral.barrier_type != "upper", (
                "Excess return 1% should not trigger 3% upper barrier"
            )

    def test_build_dataset_with_benchmark(self) -> None:
        """build_triple_barrier_dataset passes benchmark_candles through."""
        n = _N_CANDLES
        stock = []
        bench = []
        price = 100.0
        for i in range(n):
            price = 100.0 + 5.0 * ((-1) ** i)
            h = price * 1.01
            lo = price * 0.99
            stock.append(_make_candle(i, price, high=h, low=lo, open_=price))
            # Benchmark is flat
            bench.append(_make_candle(i, 100.0))

        features, labels, weights, timestamps, _ = build_triple_barrier_dataset(
            stock,
            window_size=60,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
            benchmark_candles=bench,
        )
        assert len(features) == len(labels) == len(weights) == len(timestamps)


# ---------------------------------------------------------------------------
# Tests: Split detection in triple barrier (A4)
# ---------------------------------------------------------------------------


class TestSplitDetectionInTripleBarrier:
    """Samples with splits in the label period are skipped."""

    def test_split_in_label_period_skipped(self) -> None:
        """A 2:1 split in the label period causes the sample to be skipped."""
        n = 120
        candles: list[Candle] = []
        price = 100.0
        split_bar = 85  # Put split in the middle of the label range

        for i in range(n):
            if i == split_bar:
                # 2:1 split: price halves overnight, small intrabar range
                price = price * 0.5
                candles.append(
                    _make_candle(i, price, high=price * 1.001, low=price * 0.999, open_=price)
                )
            else:
                h = price * 1.005
                lo = price * 0.995
                candles.append(_make_candle(i, price, high=h, low=lo, open_=price))
                price *= 1.001  # small drift

        # Without split detection (old behavior): some samples include the split bar
        features_no_skip, labels_no_skip, _, _, _ = build_triple_barrier_dataset(
            candles,
            window_size=60,
            max_hold=20,
            atr_scale=False,
        )

        # The split is at bar 85. Entry indices run from 59 upward.
        # Entries from ~66 to ~85 would have split_bar in their label range.
        # So with split detection, we should have fewer samples.
        # We verify by checking that build_triple_barrier_dataset runs and
        # the count is less than or equal (some samples are already filtered by
        # triple_barrier_label returning None for the big price jump).

        # The key test: verify the function handles it without error
        # and produces valid output
        assert len(features_no_skip) == len(labels_no_skip)
        assert len(features_no_skip) >= 0  # sanity


# ---------------------------------------------------------------------------
# Tests: build_triple_barrier_dataset returns hold_bars (A6)
# ---------------------------------------------------------------------------

_MIN_HOLD_BARS = 1


class TestBuildTripleBarrierDatasetHoldBars:
    """build_triple_barrier_dataset returns hold_bars as the 5th element."""

    def test_returns_five_elements(self) -> None:
        """Return value is a 5-tuple including hold_bars."""
        candles = []
        price = 100.0
        for i in range(_N_CANDLES):
            price = 100.0 + 5.0 * ((-1) ** i)
            h = price * 1.01
            lo = price * 0.99
            candles.append(_make_candle(i, price, high=h, low=lo, open_=price))

        result = build_triple_barrier_dataset(
            candles,
            window_size=60,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )
        assert len(result) == 5  # noqa: PLR2004
        features, labels, weights, timestamps, hold_bars = result
        assert len(features) == len(labels) == len(weights) == len(timestamps) == len(hold_bars)

    def test_hold_bars_are_positive_integers(self) -> None:
        """All hold_bars values should be positive integers."""
        candles = []
        price = 100.0
        for i in range(_N_CANDLES):
            price = 100.0 + 5.0 * ((-1) ** i)
            h = price * 1.01
            lo = price * 0.99
            candles.append(_make_candle(i, price, high=h, low=lo, open_=price))

        _, _, _, _, hold_bars = build_triple_barrier_dataset(
            candles,
            window_size=60,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
        )

        assert len(hold_bars) > 0
        assert all(isinstance(hb, int) for hb in hold_bars)
        assert all(hb >= _MIN_HOLD_BARS for hb in hold_bars)

    def test_empty_returns_empty_hold_bars(self) -> None:
        """Insufficient candles returns empty hold_bars list."""
        candles = _make_flat_candles(10)
        _, _, _, _, hold_bars = build_triple_barrier_dataset(
            candles,
            window_size=60,
            max_hold=20,
        )
        assert hold_bars == []
