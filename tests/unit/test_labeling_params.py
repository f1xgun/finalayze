"""Tests for parameterized ATR multipliers in triple barrier labeling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.ml.training.labeling import (
    triple_barrier_label,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)
_SYMBOL = "TEST"
_MARKET = "us"
_TF = "1d"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_candle(
    index: int,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: int = 1000,
) -> Candle:
    """Create a single candle at offset *index* days from base timestamp."""
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


def _make_volatile_candles(n: int, price: float = 100.0) -> list[Candle]:
    """Create *n* candles with ~6% daily range so ATR is computable and meaningful.

    ATR for these candles is exactly 6.0 (high-low = 6% of 100 = 6.0).
    As a fraction of price that is 0.06 (6%).
    """
    candles: list[Candle] = []
    for i in range(n):
        h = price * 1.03
        lo = price * 0.97
        candles.append(_make_candle(i, price, high=h, low=lo, open_=price))
    return candles


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpperAtrMultAffectsBarrier:
    """upper_atr_mult widens/narrows the profit-target barrier."""

    def test_high_upper_mult_prevents_upper_hit(self) -> None:
        """A large upper_atr_mult pushes the profit barrier further out.

        ATR for volatile candles = 6.0 (6% of price=100).
        With upper_atr_mult=1.0 barrier is at 100*(1+0.06) = 106.
        With upper_atr_mult=5.0 barrier is at 100*(1+0.30) = 130.
        A spike to 107 hits the narrow barrier but not the wide one.
        """
        candles = _make_volatile_candles(20, price=100.0)
        # Spike bar: high reaches 107 (7% above entry)
        candles.append(
            _make_candle(20, close=106.0, high=107.0, low=100.0, open_=100.0)
        )
        for i in range(21, 45):
            candles.append(_make_candle(i, 106.0))

        # With upper_atr_mult=1.0 barrier at 106 -> spike to 107 hits it
        result_narrow = triple_barrier_label(
            candles,
            entry_index=19,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
            upper_atr_mult=1.0,
            lower_atr_mult=5.0,  # keep lower far away
        )

        # With upper_atr_mult=5.0 barrier at 130 -> spike to 107 does NOT hit it
        result_wide = triple_barrier_label(
            candles,
            entry_index=19,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
            upper_atr_mult=5.0,
            lower_atr_mult=5.0,
        )

        # Narrow mult should trigger upper barrier
        assert result_narrow is not None
        assert result_narrow.barrier_type == "upper"

        # Wide mult should NOT trigger upper barrier (vertical or None)
        if result_wide is not None:
            assert result_wide.barrier_type != "upper"


class TestLowerAtrMultAffectsBarrier:
    """lower_atr_mult widens/narrows the stop-loss barrier."""

    def test_high_lower_mult_prevents_lower_hit(self) -> None:
        """A large lower_atr_mult pushes the stop-loss barrier further out.

        ATR for volatile candles = 6.0 (6% of price=100).
        With lower_atr_mult=1.0 barrier is at 100*(1-0.06) = 94.
        With lower_atr_mult=5.0 barrier is at 100*(1-0.30) = 70.
        A drop to 93 hits the narrow barrier but not the wide one.
        """
        candles = _make_volatile_candles(20, price=100.0)
        # Drop bar: low reaches 93 (7% below entry)
        candles.append(
            _make_candle(20, close=94.0, high=100.0, low=93.0, open_=100.0)
        )
        for i in range(21, 45):
            candles.append(_make_candle(i, 94.0))

        # With lower_atr_mult=1.0 barrier at 94 -> drop to 93 hits it
        result_narrow = triple_barrier_label(
            candles,
            entry_index=19,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
            lower_atr_mult=1.0,
            upper_atr_mult=5.0,  # keep upper far away
        )

        # With lower_atr_mult=5.0 barrier at 70 -> drop to 93 does NOT hit it
        result_wide = triple_barrier_label(
            candles,
            entry_index=19,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
            lower_atr_mult=5.0,
            upper_atr_mult=5.0,
        )

        # Narrow mult should trigger lower barrier
        assert result_narrow is not None
        assert result_narrow.barrier_type == "lower"

        # Wide mult should NOT trigger lower barrier
        if result_wide is not None:
            assert result_wide.barrier_type != "lower"


class TestAsymmetricMultipliers:
    """Different upper and lower multipliers produce asymmetric barriers."""

    def test_asymmetric_barriers_yield_expected_pnl(self) -> None:
        """Tight upper (1.0) + wide lower (5.0) makes upper barrier easier to hit.

        ATR = 6.0 (6% of 100). upper_atr_mult=1.0 -> barrier at 106.
        A spike to 107 hits the upper barrier.  lower_atr_mult=5.0 -> barrier
        at 70, so the drop to 93 on the same bar does NOT hit the lower barrier.
        """
        candles = _make_volatile_candles(20, price=100.0)
        # Move up: high=107 (hits upper at 106), low=93 (does NOT hit lower at 70)
        candles.append(
            _make_candle(20, close=106.0, high=107.0, low=93.0, open_=100.0)
        )
        for i in range(21, 45):
            candles.append(_make_candle(i, 106.0))

        result = triple_barrier_label(
            candles,
            entry_index=19,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
            upper_atr_mult=1.0,  # tight profit target: barrier at 106
            lower_atr_mult=5.0,  # wide stop loss: barrier at 70
        )

        assert result is not None
        assert result.barrier_type == "upper"
        assert result.label == 1
        assert result.pnl_pct > 0


class TestDefaultMultsMatchOldBehavior:
    """Default multipliers (2.0, 2.0) produce identical results to old code."""

    def test_defaults_unchanged(self) -> None:
        """Calling with explicit defaults matches calling without them."""
        candles = _make_volatile_candles(20, price=100.0)
        candles.append(
            _make_candle(20, close=106.0, high=106.0, low=100.0, open_=100.0)
        )
        for i in range(21, 45):
            candles.append(_make_candle(i, 106.0))

        result_implicit = triple_barrier_label(
            candles,
            entry_index=19,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
        )

        result_explicit = triple_barrier_label(
            candles,
            entry_index=19,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
            lower_atr_mult=2.0,
            upper_atr_mult=2.0,
        )

        assert result_implicit == result_explicit


class TestMultsIgnoredWhenAtrScaleOff:
    """ATR multipliers have no effect when atr_scale=False."""

    def test_atr_scale_false_ignores_mults(self) -> None:
        candles = _make_volatile_candles(20, price=100.0)
        candles.append(
            _make_candle(20, close=104.0, high=104.0, low=100.0, open_=100.0)
        )
        for i in range(21, 45):
            candles.append(_make_candle(i, 104.0))

        result_a = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
            upper_atr_mult=1.0,
            lower_atr_mult=1.0,
        )

        result_b = triple_barrier_label(
            candles,
            entry_index=19,
            upper_pct=0.03,
            lower_pct=0.03,
            max_hold=20,
            atr_scale=False,
            upper_atr_mult=10.0,
            lower_atr_mult=10.0,
        )

        # Both should be identical because atr_scale=False
        assert result_a == result_b


class TestPnlReflectsMultiplier:
    """The reported pnl_pct matches the ATR-scaled barrier placement."""

    def test_upper_pnl_scales_with_mult(self) -> None:
        """Tighter upper_atr_mult produces smaller pnl_pct on upper barrier hit."""
        candles = _make_volatile_candles(20, price=100.0)
        # Big spike so both mults hit upper
        candles.append(
            _make_candle(20, close=120.0, high=120.0, low=100.0, open_=100.0)
        )
        for i in range(21, 45):
            candles.append(_make_candle(i, 120.0))

        result_tight = triple_barrier_label(
            candles,
            entry_index=19,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
            upper_atr_mult=1.0,
            lower_atr_mult=5.0,
        )

        result_loose = triple_barrier_label(
            candles,
            entry_index=19,
            max_hold=20,
            atr_scale=True,
            atr_period=14,
            upper_atr_mult=3.0,
            lower_atr_mult=5.0,
        )

        assert result_tight is not None
        assert result_loose is not None
        assert result_tight.barrier_type == "upper"
        assert result_loose.barrier_type == "upper"
        # Tighter mult -> barrier is closer -> smaller pnl
        assert result_tight.pnl_pct < result_loose.pnl_pct
