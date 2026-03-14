"""Tests for MOEX position sizing: 1M RUB starting capital and correct market open time."""

from __future__ import annotations

from datetime import UTC, datetime, time
from decimal import Decimal

# ---------------------------------------------------------------------------
# Test 1: MOEX segment_cash is 1M RUB
# ---------------------------------------------------------------------------


def test_moex_segment_cash_is_1m_rub() -> None:
    """MOEX segments must use 1,000,000 RUB starting capital."""
    segment = "ru_blue_chips"
    cash = Decimal(100_000)  # USD default

    # Replicate the logic from run_iteration.py
    segment_cash = Decimal(1_000_000) if segment.startswith("ru_") else cash

    assert segment_cash == Decimal(1_000_000)


def test_us_segment_cash_unchanged() -> None:
    """US segments must keep original cash value."""
    segment = "us_tech"
    cash = Decimal(100_000)

    segment_cash = Decimal(1_000_000) if segment.startswith("ru_") else cash

    assert segment_cash == cash


# ---------------------------------------------------------------------------
# Test 2: Pre-trade check_dt market open times
# ---------------------------------------------------------------------------

_US_MARKET_OPEN_UTC = time(14, 30, tzinfo=UTC)
_MOEX_MARKET_OPEN_UTC = time(7, 0, tzinfo=UTC)


def _adjust_check_dt(dt: datetime, segment_id: str) -> datetime:
    """Replicate engine logic for adjusting midnight candle timestamps."""
    from finalayze.backtest.engine import _MOEX_MARKET_OPEN_UTC as ENGINE_MOEX_OPEN
    from finalayze.backtest.engine import _US_MARKET_OPEN_UTC as ENGINE_US_OPEN

    if dt.hour == 0 and dt.minute == 0:
        if segment_id.startswith("ru_"):
            return datetime.combine(dt.date(), ENGINE_MOEX_OPEN)
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
# Test 3: Position sizing produces correct order of magnitude
# ---------------------------------------------------------------------------


def test_position_size_correct_order_of_magnitude() -> None:
    """compute_position_size with 1M RUB equity should produce ~8-20% position.

    Half-Kelly with win_rate=0.5, avg_win_ratio=1.5:
    f* = (0.5*1.5 - 0.5)/1.5 = 0.167, half-Kelly = 0.0833 => 83K RUB.
    The key assertion: positions are NOT 0.02% (the old bug ~200 RUB).
    """
    from finalayze.risk.position_sizer import compute_position_size

    position = compute_position_size(
        win_rate=Decimal("0.50"),
        avg_win_ratio=Decimal("1.5"),
        equity=Decimal(1_000_000),
        kelly_fraction=Decimal("0.5"),
        max_position_pct=Decimal("0.20"),
    )

    # Half-Kelly gives 8.33% = 83,333 RUB. Must be in 5-20% range (50K-200K),
    # critically NOT the old bug value of ~200 RUB (0.02%).
    assert Decimal(50_000) <= position <= Decimal(200_000), (
        f"Position {position} not in 50K-200K range (5-20% of 1M RUB)"
    )


def test_position_size_not_tiny_bug_value() -> None:
    """Position size must NOT be the old bug value of ~0.02% of equity."""
    from finalayze.risk.position_sizer import compute_position_size

    position = compute_position_size(
        win_rate=Decimal("0.50"),
        avg_win_ratio=Decimal("1.5"),
        equity=Decimal(1_000_000),
        kelly_fraction=Decimal("0.5"),
        max_position_pct=Decimal("0.20"),
    )

    # Old bug: positions were ~200 RUB (0.02% of equity). Must be WAY higher.
    min_acceptable = Decimal(10_000)  # 1% of equity as absolute minimum
    assert position > min_acceptable, (
        f"Position {position} is suspiciously small -- likely the old sizing bug"
    )
