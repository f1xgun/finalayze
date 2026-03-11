"""Tests for MultiTimeframeContext builder (Task 4.1).

Validates:
- Weekly / monthly bar aggregation with completed-period-only protocol
- 2-bar external data lag to prevent look-ahead bias
- RSI computation (Wilder's smoothing)
- SMA-50 ratio on weekly closes
- Monthly trend direction (+1, 0, -1)
- Full integration pipeline
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, MultiTimeframeContext
from finalayze.ml.features.multi_timeframe import (
    _EXTERNAL_DATA_LAG_BARS,
    _compute_rsi,
    aggregate_monthly_bars,
    aggregate_weekly_bars,
    build_multi_timeframe_context,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RSI_MIN_PERIOD = 14
_RSI_MAX_VALUE = 100.0
_RSI_MIN_VALUE = 0.0
_SMA_50_WEEKS = 50
_TREND_MONTHS_NEEDED = 3


def _make_daily_candle(
    dt: date,
    price: float = 100.0,
    volume: int = 1000,
    symbol: str = "TEST",
    market_id: str = "us",
) -> Candle:
    """Create a single daily candle for a given date."""
    ts = datetime(dt.year, dt.month, dt.day, 16, 0, tzinfo=UTC)
    return Candle(
        symbol=symbol,
        market_id=market_id,
        timeframe="1d",
        timestamp=ts,
        open=Decimal(str(price - 0.5)),
        high=Decimal(str(price + 1.0)),
        low=Decimal(str(price - 1.0)),
        close=Decimal(str(price)),
        volume=volume,
    )


def _make_weekday_candles(
    start: date,
    n_days: int,
    base_price: float = 100.0,
    price_step: float = 0.5,
) -> list[Candle]:
    """Create n_days worth of candles, skipping weekends."""
    candles: list[Candle] = []
    current = start
    count = 0
    while count < n_days:
        if current.weekday() < 5:  # Mon-Fri
            price = base_price + count * price_step
            candles.append(_make_daily_candle(current, price=price))
            count += 1
        current += timedelta(days=1)
    return candles


# ---------------------------------------------------------------------------
# Weekly aggregation tests
# ---------------------------------------------------------------------------


class TestWeeklyAggregation:
    """Test weekly bar aggregation from daily candles."""

    def test_weekly_aggregation_basic(self) -> None:
        """Aggregate 10+ business days into weekly bars."""
        # Start Monday 2025-01-06, generate 15 trading days (3 weeks)
        candles = _make_weekday_candles(date(2025, 1, 6), n_days=15)
        # current_date well after the 3 weeks + lag
        current = date(2025, 2, 1)
        weekly = aggregate_weekly_bars(candles, current)
        assert len(weekly) >= 2  # at least 2 completed weeks

        # First weekly bar should have OHLCV aggregated
        bar = weekly[0]
        assert bar.timeframe == "1w"
        assert bar.volume > 0

    def test_weekly_excludes_current_incomplete_week(self) -> None:
        """Current week's bars must NOT be included."""
        # 3 weeks of data: Mon 2025-01-06 through Fri 2025-01-24
        candles = _make_weekday_candles(date(2025, 1, 6), n_days=15)
        # Add 2 days in the next week (Mon-Tue 2025-01-27-28)
        candles += _make_weekday_candles(date(2025, 1, 27), n_days=2, base_price=120.0)

        # current_date = Wednesday 2025-01-29 (within the partial 4th week)
        # Even with lag=0 hypothetically, the 4th week is incomplete
        # With lag=2, the 3rd week (ended 2025-01-24) needs 2 biz days after Friday
        # That's Tuesday 2025-01-28. So on Wed 2025-01-29, the 3rd week IS available.
        current = date(2025, 1, 29)
        weekly = aggregate_weekly_bars(candles, current)

        # The partial 4th week (Mon-Tue 2025-01-27-28) must NOT appear
        for bar in weekly:
            bar_date = bar.timestamp.date()
            assert bar_date < date(2025, 1, 27), (
                f"Partial week bar {bar_date} should not be included"
            )

    def test_weekly_respects_lag(self) -> None:
        """Completed week not available until 2 business days after Friday."""
        # Week of 2025-01-06 to 2025-01-10 (Mon-Fri)
        candles = _make_weekday_candles(date(2025, 1, 6), n_days=5)

        # On Monday 2025-01-13: only 1 biz day after Friday -> NOT available
        weekly_mon = aggregate_weekly_bars(candles, date(2025, 1, 13))
        assert len(weekly_mon) == 0

        # On Tuesday 2025-01-14: 2 biz days after Friday -> available
        weekly_tue = aggregate_weekly_bars(candles, date(2025, 1, 14))
        assert len(weekly_tue) == 1

    def test_weekly_ohlcv_aggregation(self) -> None:
        """O=first open, H=max high, L=min low, C=last close, V=sum volume."""
        candles = [
            _make_daily_candle(date(2025, 1, 6), price=100.0, volume=100),
            _make_daily_candle(date(2025, 1, 7), price=105.0, volume=200),
            _make_daily_candle(date(2025, 1, 8), price=95.0, volume=150),
            _make_daily_candle(date(2025, 1, 9), price=102.0, volume=120),
            _make_daily_candle(date(2025, 1, 10), price=103.0, volume=180),
        ]
        # Well after lag
        current = date(2025, 2, 1)
        weekly = aggregate_weekly_bars(candles, current)
        assert len(weekly) == 1

        bar = weekly[0]
        # O = first day's open = 100.0 - 0.5 = 99.5
        assert bar.open == Decimal("99.5")
        # H = max high = 105.0 + 1.0 = 106.0
        assert bar.high == Decimal("106.0")
        # L = min low = 95.0 - 1.0 = 94.0
        assert bar.low == Decimal("94.0")
        # C = last day's close = 103.0
        assert bar.close == Decimal("103.0")
        # V = sum = 100 + 200 + 150 + 120 + 180 = 750
        assert bar.volume == 750


# ---------------------------------------------------------------------------
# Monthly aggregation tests
# ---------------------------------------------------------------------------


class TestMonthlyAggregation:
    """Test monthly bar aggregation from daily candles."""

    def test_monthly_aggregation(self) -> None:
        """Basic monthly bar creation from daily candles."""
        # Full January 2025 (23 trading days) + some February
        candles = _make_weekday_candles(date(2025, 1, 2), n_days=30)
        # Well after January end + lag
        current = date(2025, 3, 1)
        monthly = aggregate_monthly_bars(candles, current)
        assert len(monthly) >= 1

        bar = monthly[0]
        assert bar.timeframe == "1M"

    def test_monthly_excludes_partial_month(self) -> None:
        """Current month's bars must NOT be included."""
        # January 2025 data
        candles = _make_weekday_candles(date(2025, 1, 2), n_days=22)
        # Add some February data
        candles += _make_weekday_candles(date(2025, 2, 3), n_days=5, base_price=120.0)

        # current_date in mid-February: February is partial
        current = date(2025, 2, 15)
        monthly = aggregate_monthly_bars(candles, current)

        # February bars should not appear as a monthly bar
        for bar in monthly:
            bar_month = bar.timestamp.date().month
            assert bar_month != 2, "Partial February should not be included"  # noqa: PLR2004

    def test_monthly_respects_lag(self) -> None:
        """Completed month not available until 2 business days after month end."""
        # Full January 2025
        candles = _make_weekday_candles(date(2025, 1, 2), n_days=22)

        # Jan 31 is Friday. 2 biz days after = Tue Feb 4
        # On Monday Feb 3: only 1 biz day after month end -> NOT available
        monthly_early = aggregate_monthly_bars(candles, date(2025, 2, 3))
        assert len(monthly_early) == 0

        # On Tuesday Feb 4: 2 biz days after -> available
        monthly_ok = aggregate_monthly_bars(candles, date(2025, 2, 4))
        assert len(monthly_ok) == 1


# ---------------------------------------------------------------------------
# RSI computation tests
# ---------------------------------------------------------------------------


class TestRSIComputation:
    """Test Wilder's RSI formula."""

    def test_rsi_computation(self) -> None:
        """RSI result should be in [0, 100] range."""
        # Uptrend: steady increase
        closes = [100.0 + i * 0.5 for i in range(20)]
        rsi = _compute_rsi(closes, period=_RSI_MIN_PERIOD)
        assert rsi is not None
        assert _RSI_MIN_VALUE <= rsi <= _RSI_MAX_VALUE

    def test_rsi_insufficient_data(self) -> None:
        """Returns None when fewer than period+1 closes."""
        closes = [100.0 + i for i in range(10)]
        rsi = _compute_rsi(closes, period=_RSI_MIN_PERIOD)
        assert rsi is None

    def test_rsi_all_up(self) -> None:
        """Monotonically increasing closes should yield RSI near 100."""
        closes = [100.0 + i * 1.0 for i in range(30)]
        rsi = _compute_rsi(closes, period=_RSI_MIN_PERIOD)
        assert rsi is not None
        rsi_high_threshold = 90.0
        assert rsi > rsi_high_threshold

    def test_rsi_all_down(self) -> None:
        """Monotonically decreasing closes should yield RSI near 0."""
        closes = [200.0 - i * 1.0 for i in range(30)]
        rsi = _compute_rsi(closes, period=_RSI_MIN_PERIOD)
        assert rsi is not None
        rsi_low_threshold = 10.0
        assert rsi < rsi_low_threshold


# ---------------------------------------------------------------------------
# SMA-50 ratio tests
# ---------------------------------------------------------------------------


class TestSMA50Ratio:
    """Test SMA-50 ratio computation on weekly closes."""

    def test_sma50_ratio(self) -> None:
        """SMA-50 ratio = close / SMA(50) on weekly closes."""
        # Generate enough data for 50+ completed weeks
        # 50 weeks = 250 trading days, + lag buffer
        candles = _make_weekday_candles(date(2024, 1, 2), n_days=270)
        current = date(2025, 3, 1)
        ctx = build_multi_timeframe_context(candles, current)
        assert ctx.weekly_sma_50_ratio is not None
        # With steadily increasing prices, ratio > 1
        assert ctx.weekly_sma_50_ratio > 0

    def test_sma50_insufficient_data(self) -> None:
        """Returns None when < 50 weekly bars available."""
        # Only 10 weeks of data
        candles = _make_weekday_candles(date(2025, 1, 6), n_days=50)
        current = date(2025, 6, 1)
        ctx = build_multi_timeframe_context(candles, current)
        assert ctx.weekly_sma_50_ratio is None


# ---------------------------------------------------------------------------
# Monthly trend direction tests
# ---------------------------------------------------------------------------


class TestMonthlyTrendDirection:
    """Test monthly trend direction computation."""

    def test_monthly_trend_up(self) -> None:
        """3 ascending monthly closes -> +1."""
        # Build 4+ months of data with ascending prices
        candles: list[Candle] = []
        for month_offset in range(5):
            month = 1 + month_offset
            year = 2025
            if month > 12:
                month -= 12
                year += 1
            start = date(year, month, 2)
            price = 100.0 + month_offset * 10.0
            candles += _make_weekday_candles(start, n_days=20, base_price=price)

        current = date(2025, 8, 1)
        ctx = build_multi_timeframe_context(candles, current)
        assert ctx.monthly_trend_direction == 1

    def test_monthly_trend_down(self) -> None:
        """3 descending monthly closes -> -1."""
        candles: list[Candle] = []
        for month_offset in range(5):
            month = 1 + month_offset
            year = 2025
            if month > 12:
                month -= 12
                year += 1
            start = date(year, month, 2)
            price = 200.0 - month_offset * 20.0
            candles += _make_weekday_candles(start, n_days=20, base_price=price)

        current = date(2025, 8, 1)
        ctx = build_multi_timeframe_context(candles, current)
        assert ctx.monthly_trend_direction == -1

    def test_monthly_trend_neutral(self) -> None:
        """Mixed direction -> 0."""
        candles: list[Candle] = []
        prices = [100.0, 110.0, 105.0, 115.0, 108.0]
        for month_offset in range(5):
            month = 1 + month_offset
            start = date(2025, month, 2)
            candles += _make_weekday_candles(
                start,
                n_days=20,
                base_price=prices[month_offset],
            )

        current = date(2025, 8, 1)
        ctx = build_multi_timeframe_context(candles, current)
        assert ctx.monthly_trend_direction == 0

    def test_monthly_trend_insufficient(self) -> None:
        """< 3 completed months -> None."""
        # Only ~1 month of data
        candles = _make_weekday_candles(date(2025, 1, 2), n_days=22)
        current = date(2025, 3, 1)
        ctx = build_multi_timeframe_context(candles, current)
        assert ctx.monthly_trend_direction is None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestBuildContextIntegration:
    """Full pipeline integration tests."""

    def test_build_context_integration(self) -> None:
        """Full pipeline: daily candles -> MultiTimeframeContext."""
        # ~6 months of daily data
        candles = _make_weekday_candles(date(2024, 7, 1), n_days=130)
        current = date(2025, 2, 1)
        ctx = build_multi_timeframe_context(candles, current)

        assert isinstance(ctx, MultiTimeframeContext)
        # Should have weekly and monthly bars
        assert ctx.weekly_completed is not None
        assert ctx.monthly_completed is not None
        # Weekly RSI should be computed (enough weekly bars)
        assert ctx.weekly_rsi_14 is not None

    def test_empty_candles(self) -> None:
        """Empty input -> all None fields."""
        ctx = build_multi_timeframe_context([], date(2025, 1, 1))
        assert ctx.weekly_completed is None
        assert ctx.monthly_completed is None
        assert ctx.weekly_rsi_14 is None
        assert ctx.weekly_sma_50_ratio is None
        assert ctx.monthly_trend_direction is None

    def test_no_look_ahead_bias(self) -> None:
        """Candles after current_date must be excluded."""
        candles = _make_weekday_candles(date(2025, 1, 6), n_days=60)
        # Use a current_date that is in the middle of the data
        mid_date = date(2025, 2, 1)

        ctx = build_multi_timeframe_context(candles, mid_date)

        # Build again with only candles up to mid_date
        filtered = [c for c in candles if c.timestamp.date() < mid_date]
        ctx_filtered = build_multi_timeframe_context(filtered, mid_date)

        # Results should be identical -- future candles were excluded
        assert ctx.weekly_completed == ctx_filtered.weekly_completed
        assert ctx.monthly_completed == ctx_filtered.monthly_completed
        assert ctx.weekly_rsi_14 == ctx_filtered.weekly_rsi_14
        assert ctx.weekly_sma_50_ratio == ctx_filtered.weekly_sma_50_ratio
        assert ctx.monthly_trend_direction == ctx_filtered.monthly_trend_direction

    def test_frozen_dataclass(self) -> None:
        """MultiTimeframeContext should be frozen (immutable)."""
        ctx = build_multi_timeframe_context([], date(2025, 1, 1))
        with pytest.raises(AttributeError):
            ctx.weekly_rsi_14 = 50.0  # type: ignore[misc]

    def test_external_data_lag_constant(self) -> None:
        """Verify the lag constant is 2."""
        expected_lag = 2
        assert expected_lag == _EXTERNAL_DATA_LAG_BARS
