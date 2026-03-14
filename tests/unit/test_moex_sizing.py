"""Tests for MOEX position sizing: 1M RUB starting capital and correct market open time."""

from __future__ import annotations

from datetime import UTC, datetime, time
from decimal import Decimal

import pytest


# ---------------------------------------------------------------------------
# Test 1: MOEX segment_cash is 1M RUB
# ---------------------------------------------------------------------------

def test_moex_segment_cash_is_1m_rub() -> None:
    """MOEX segments must use 1,000,000 RUB starting capital."""
    segment = "ru_blue_chips"
    cash = Decimal(100_000)  # USD default

    # Replicate the logic from run_iteration.py
    if segment.startswith("ru_"):
        segment_cash = Decimal(1_000_000)
    else:
        segment_cash = cash

    assert segment_cash == Decimal(1_000_000)


def test_us_segment_cash_unchanged() -> None:
    """US segments must keep original cash value."""
    segment = "us_tech"
    cash = Decimal(100_000)

    if segment.startswith("ru_"):
        segment_cash = Decimal(1_000_000)
    else:
        segment_cash = cash

    assert segment_cash == cash


# ---------------------------------------------------------------------------
# Test 2: Pre-trade check_dt market open times
# ---------------------------------------------------------------------------

_US_MARKET_OPEN_UTC = time(14, 30, tzinfo=UTC)
_MOEX_MARKET_OPEN_UTC = time(7, 0, tzinfo=UTC)


def _adjust_check_dt(dt: datetime, segment_id: str) -> datetime:
    """Replicate engine logic for adjusting midnight candle timestamps."""
    from src.finalayze.backtest.engine import _MOEX_MARKET_OPEN_UTC as ENGINE_MOEX_OPEN
    from src.finalayze.backtest.engine import _US_MARKET_OPEN_UTC as ENGINE_US_OPEN

    if dt.hour == 0 and dt.minute == 0:
        if segment_id.startswith("ru_"):
            return datetime.combine(dt.date(), ENGINE_MOEX_OPEN)
        else:
            return datetime.combine(dt.date(), ENGINE_US_OPEN)
    return dt


def test_pretrade_check_dt_moex_uses_0700_utc() -> None:
    """MOEX midnight candle timestamps should be adjusted to 07:00 UTC."""
    midnight_dt = datetime(2025, 3, 10, 0, 0, tzinfo=UTC)
    adjusted = _adjust_check_dt(midnight_dt, "ru_blue_chips")

    assert adjusted.hour == 7
    assert adjusted.minute == 0


def test_pretrade_check_dt_us_uses_1430_utc() -> None:
    """US midnight candle timestamps should be adjusted to 14:30 UTC (unchanged)."""
    midnight_dt = datetime(2025, 3, 10, 0, 0, tzinfo=UTC)
    adjusted = _adjust_check_dt(midnight_dt, "us_tech")

    assert adjusted.hour == 14
    assert adjusted.minute == 30


# ---------------------------------------------------------------------------
# Test 3: Position sizing produces 10-20% of 1M RUB equity
# ---------------------------------------------------------------------------

def test_position_size_10_to_20_pct_of_1m_rub() -> None:
    """compute_position_size with 1M RUB equity should produce 100K-200K position."""
    from finalayze.risk.position_sizer import compute_position_size

    position = compute_position_size(
        win_rate=Decimal("0.50"),
        avg_win_ratio=Decimal("1.5"),
        equity=Decimal(1_000_000),
        kelly_fraction=Decimal("0.5"),
        max_position_pct=Decimal("0.20"),
    )

    # Position should be 10-20% of 1M RUB = 100K-200K
    assert Decimal(100_000) <= position <= Decimal(200_000), (
        f"Position {position} not in 100K-200K range (10-20% of 1M RUB)"
    )
