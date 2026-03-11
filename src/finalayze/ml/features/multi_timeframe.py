"""Multi-timeframe context builder (Layer 3).

Aggregates daily candles into completed weekly / monthly bars and derives
higher-timeframe features (RSI-14, SMA-50 ratio, monthly trend direction).

All values use COMPLETED periods only (no partial bars).
A 2-bar lag (_EXTERNAL_DATA_LAG_BARS) is applied on top to prevent look-ahead bias.
"""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta

from finalayze.core.schemas import Candle, MultiTimeframeContext

_EXTERNAL_DATA_LAG_BARS = 2  # 2 daily bars lag to prevent look-ahead

_RSI_PERIOD = 14
_SMA_WEEKS = 50
_TREND_MONTHS = 3
_WEEKDAYS_IN_WEEK = 5  # Mon-Fri


def _business_days_after(reference: date, current_date: date) -> int:
    """Count business days (Mon-Fri) from reference (exclusive) up to current_date (inclusive).

    Example: reference=Friday, current_date=Tuesday -> Monday(1) + Tuesday(2) = 2.
    """
    if current_date <= reference:
        return 0
    count = 0
    day = reference + timedelta(days=1)
    while day <= current_date:
        if day.weekday() < _WEEKDAYS_IN_WEEK:
            count += 1
        day += timedelta(days=1)
    return count


def _last_day_of_month(year: int, month: int) -> date:
    """Return the last calendar day of the given month."""
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, last_day)


def aggregate_weekly_bars(
    daily_candles: list[Candle],
    current_date: date,
) -> list[Candle]:
    """Aggregate daily candles into completed weekly bars (Mon-Fri).

    A week is 'completed' if current_date is at least 2 business days after the Friday.
    Returns list of completed weekly Candle objects (OHLCV aggregated).
    """
    if not daily_candles:
        return []

    # Group candles by ISO week (year, week_number)
    weeks: dict[tuple[int, int], list[Candle]] = {}
    for c in daily_candles:
        if c.timestamp.date() >= current_date:
            continue  # Exclude future candles
        iso_year, iso_week, _ = c.timestamp.date().isocalendar()
        key = (iso_year, iso_week)
        if key not in weeks:
            weeks[key] = []
        weeks[key].append(c)

    # Determine which weeks are completed with lag
    completed: list[Candle] = []
    for (iso_year, iso_week), candles in sorted(weeks.items()):
        # Find the Friday of this ISO week
        # ISO week 1 always contains Jan 4, Monday is day 1
        jan4 = date(iso_year, 1, 4)
        # Monday of ISO week 1
        mon_wk1 = jan4 - timedelta(days=jan4.weekday())
        # Monday of the target week
        monday = mon_wk1 + timedelta(weeks=iso_week - 1)
        friday = monday + timedelta(days=4)

        # Check if current_date is at least _EXTERNAL_DATA_LAG_BARS business days
        # after Friday
        biz_days_after = _business_days_after(friday, current_date)
        if biz_days_after < _EXTERNAL_DATA_LAG_BARS:
            continue

        # Also ensure the week is truly in the past (current_date > friday)
        if current_date <= friday:
            continue

        # Check that the current week (containing current_date) is not this week
        cur_iso_year, cur_iso_week, _ = current_date.isocalendar()
        if (iso_year, iso_week) == (cur_iso_year, cur_iso_week):
            continue

        # Aggregate OHLCV
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        bar = _aggregate_candles(sorted_candles, timeframe="1w", bar_date=friday)
        completed.append(bar)

    return completed


def aggregate_monthly_bars(
    daily_candles: list[Candle],
    current_date: date,
) -> list[Candle]:
    """Aggregate daily candles into completed monthly bars.

    A month is 'completed' if current_date is at least 2 business days after month end.
    """
    if not daily_candles:
        return []

    # Group candles by (year, month)
    months: dict[tuple[int, int], list[Candle]] = {}
    for c in daily_candles:
        if c.timestamp.date() >= current_date:
            continue
        key = (c.timestamp.year, c.timestamp.month)
        if key not in months:
            months[key] = []
        months[key].append(c)

    completed: list[Candle] = []
    for (year, month), candles in sorted(months.items()):
        month_end = _last_day_of_month(year, month)

        # Check that current_date is past the month end
        if current_date <= month_end:
            continue

        # Check that current month is not the same
        if (current_date.year, current_date.month) == (year, month):
            continue

        # Check lag: at least _EXTERNAL_DATA_LAG_BARS biz days after month end
        biz_days_after = _business_days_after(month_end, current_date)
        if biz_days_after < _EXTERNAL_DATA_LAG_BARS:
            continue

        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        bar = _aggregate_candles(sorted_candles, timeframe="1M", bar_date=month_end)
        completed.append(bar)

    return completed


def _aggregate_candles(
    candles: list[Candle],
    timeframe: str,
    bar_date: date,
) -> Candle:
    """Aggregate a list of sorted daily candles into a single bar.

    O=first open, H=max high, L=min low, C=last close, V=sum volume.
    """
    first = candles[0]
    return Candle(
        symbol=first.symbol,
        market_id=first.market_id,
        timeframe=timeframe,
        timestamp=datetime(bar_date.year, bar_date.month, bar_date.day, 23, 59, tzinfo=UTC),
        open=candles[0].open,
        high=max(c.high for c in candles),
        low=min(c.low for c in candles),
        close=candles[-1].close,
        volume=sum(c.volume for c in candles),
    )


def _compute_rsi(closes: list[float], period: int = _RSI_PERIOD) -> float | None:
    """Compute RSI using Wilder's smoothed moving average.

    Args:
        closes: List of closing prices.
        period: RSI period (default 14).

    Returns:
        RSI value in [0, 100], or None if insufficient data.
    """
    min_required = period + 1
    if len(closes) < min_required:
        return None

    # Compute price changes
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Initial average gain/loss over first `period` changes
    gains = [max(d, 0.0) for d in deltas[:period]]
    losses = [max(-d, 0.0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder's smoothing for remaining changes
    for d in deltas[period:]:
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    rsi_ceiling = 100.0

    if avg_loss == 0.0:
        return rsi_ceiling

    rs = avg_gain / avg_loss
    return rsi_ceiling - rsi_ceiling / (1.0 + rs)


def _compute_sma_50_ratio(weekly_closes: list[float]) -> float | None:
    """Compute close / SMA(50) ratio on weekly closes.

    Returns None if fewer than 50 weekly closes.
    """
    if len(weekly_closes) < _SMA_WEEKS:
        return None

    sma_50 = sum(weekly_closes[-_SMA_WEEKS:]) / _SMA_WEEKS
    if sma_50 <= 0.0:
        return None

    return weekly_closes[-1] / sma_50


def _compute_monthly_trend(monthly_closes: list[float]) -> int | None:
    """Compute monthly trend direction from the last 3 monthly closes.

    Returns:
        +1 if all 3 ascending, -1 if all 3 descending, 0 otherwise.
        None if fewer than 3 monthly closes.
    """
    if len(monthly_closes) < _TREND_MONTHS:
        return None

    last_three = monthly_closes[-_TREND_MONTHS:]

    # All ascending: each > previous
    all_up = all(last_three[i] > last_three[i - 1] for i in range(1, _TREND_MONTHS))
    if all_up:
        return 1

    # All descending: each < previous
    all_down = all(last_three[i] < last_three[i - 1] for i in range(1, _TREND_MONTHS))
    if all_down:
        return -1

    return 0


def build_multi_timeframe_context(
    daily_candles: list[Candle],
    current_date: date,
    symbol: str = "UNKNOWN",  # noqa: ARG001
    market_id: str = "us",  # noqa: ARG001
) -> MultiTimeframeContext:
    """Build MultiTimeframeContext from daily candles at a given date.

    Steps:
    1. Filter daily_candles to only those before current_date
    2. Aggregate to weekly/monthly completed bars (with 2-bar lag)
    3. Compute weekly RSI-14 from weekly closes
    4. Compute weekly SMA-50 ratio (close / SMA50)
    5. Compute monthly trend direction: +1 if last 3 months all up, -1 if all down, 0 else
    """
    if not daily_candles:
        return MultiTimeframeContext()

    # Step 1: filter to past data only
    filtered = [c for c in daily_candles if c.timestamp.date() < current_date]
    if not filtered:
        return MultiTimeframeContext()

    # Step 2: aggregate
    weekly_bars = aggregate_weekly_bars(filtered, current_date)
    monthly_bars = aggregate_monthly_bars(filtered, current_date)

    # Last completed bars
    weekly_completed = weekly_bars[-1] if weekly_bars else None
    monthly_completed = monthly_bars[-1] if monthly_bars else None

    # Step 3: weekly RSI-14
    weekly_closes = [float(b.close) for b in weekly_bars]
    weekly_rsi_14 = _compute_rsi(weekly_closes, period=_RSI_PERIOD)

    # Step 4: weekly SMA-50 ratio
    weekly_sma_50_ratio = _compute_sma_50_ratio(weekly_closes)

    # Step 5: monthly trend direction
    monthly_closes = [float(b.close) for b in monthly_bars]
    monthly_trend_direction = _compute_monthly_trend(monthly_closes)

    return MultiTimeframeContext(
        weekly_completed=weekly_completed,
        monthly_completed=monthly_completed,
        weekly_rsi_14=weekly_rsi_14,
        weekly_sma_50_ratio=weekly_sma_50_ratio,
        monthly_trend_direction=monthly_trend_direction,
    )
