"""Unit tests for MarketSchedule trading hours (Layer 2)."""

from __future__ import annotations

from datetime import UTC, datetime

from finalayze.markets.schedule import (
    MOEX_MARKET_SCHEDULE,
    SCHEDULES,
    US_MARKET_SCHEDULE,
    MarketSchedule,
)

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────

# Monday 2024-01-15:
#   14:00 UTC = 09:00 ET  -> US market not yet open (opens 09:30)
#   14:31 UTC = 09:31 ET  -> US market open (after 09:30)
#   15:00 UTC = 10:00 ET  -> US market open
#   21:00 UTC = 16:00 ET  -> US market boundary (close is 16:00, exclusive)
#   21:30 UTC = 16:30 ET  -> US market closed (after 16:00)

US_BEFORE_OPEN_UTC = datetime(2024, 1, 15, 14, 0, tzinfo=UTC)  # 09:00 ET
US_EXACT_OPEN_UTC = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)  # 09:30 ET (inclusive lower bound)
US_ONE_MIN_AFTER_OPEN_UTC = datetime(2024, 1, 15, 14, 31, tzinfo=UTC)  # 09:31 ET
US_DURING_HOURS_UTC = datetime(2024, 1, 15, 15, 0, tzinfo=UTC)  # 10:00 ET
US_AT_CLOSE_UTC = datetime(2024, 1, 15, 21, 0, tzinfo=UTC)  # 16:00 ET (closed)
US_AFTER_CLOSE_UTC = datetime(2024, 1, 15, 21, 30, tzinfo=UTC)  # 16:30 ET

# Weekend dates for weekday guard tests:
#   2024-01-13 (Saturday) 14:31 UTC = 09:31 ET -- within trading hours but weekend
#   2024-01-14 (Sunday)   14:31 UTC = 09:31 ET -- within trading hours but weekend
US_SATURDAY_UTC = datetime(2024, 1, 13, 14, 31, tzinfo=UTC)  # Saturday 09:31 ET
US_SUNDAY_UTC = datetime(2024, 1, 14, 14, 31, tzinfo=UTC)  # Sunday 09:31 ET

# Friday 2024-01-19 after-close: next_open must skip to Monday 2024-01-22, not Saturday
US_FRIDAY_AFTER_CLOSE_UTC = datetime(2024, 1, 19, 21, 30, tzinfo=UTC)  # Friday 16:30 ET

# Monday 2024-01-15 for MOEX:
#   07:00 UTC = 10:00 MSK -> MOEX open
#   15:45 UTC = 18:45 MSK -> MOEX closed (after 18:40)
#   06:59 UTC = 09:59 MSK -> MOEX not yet open

MOEX_BEFORE_OPEN_UTC = datetime(2024, 1, 15, 6, 59, tzinfo=UTC)  # 09:59 MSK
MOEX_DURING_HOURS_UTC = datetime(2024, 1, 15, 7, 0, tzinfo=UTC)  # 10:00 MSK
MOEX_AFTER_CLOSE_UTC = datetime(2024, 1, 15, 15, 45, tzinfo=UTC)  # 18:45 MSK

EXPECTED_SCHEDULE_COUNT = 2


# ── MarketSchedule ───────────────────────────────────────────────────────


class TestMarketSchedule:
    def test_is_market_open_during_us_hours(self) -> None:
        assert US_MARKET_SCHEDULE.is_market_open(US_DURING_HOURS_UTC) is True

    def test_is_market_open_at_us_open_time(self) -> None:
        assert US_MARKET_SCHEDULE.is_market_open(US_ONE_MIN_AFTER_OPEN_UTC) is True

    def test_is_market_open_at_exact_open_boundary(self) -> None:
        # 09:30 ET is inclusive lower bound — market is open at exactly open time
        assert US_MARKET_SCHEDULE.is_market_open(US_EXACT_OPEN_UTC) is True

    def test_is_market_closed_on_saturday(self) -> None:
        # Saturday: even during normal trading hours the market is closed
        assert US_MARKET_SCHEDULE.is_market_open(US_SATURDAY_UTC) is False

    def test_is_market_closed_on_sunday(self) -> None:
        # Sunday: even during normal trading hours the market is closed
        assert US_MARKET_SCHEDULE.is_market_open(US_SUNDAY_UTC) is False

    def test_is_market_closed_before_us_open(self) -> None:
        assert US_MARKET_SCHEDULE.is_market_open(US_BEFORE_OPEN_UTC) is False

    def test_is_market_closed_at_us_close_time(self) -> None:
        # Close time is exclusive: 16:00 ET means closed at or after 16:00
        assert US_MARKET_SCHEDULE.is_market_open(US_AT_CLOSE_UTC) is False

    def test_is_market_closed_after_us_close(self) -> None:
        assert US_MARKET_SCHEDULE.is_market_open(US_AFTER_CLOSE_UTC) is False

    def test_is_market_open_during_moex_hours(self) -> None:
        assert MOEX_MARKET_SCHEDULE.is_market_open(MOEX_DURING_HOURS_UTC) is True

    def test_is_market_closed_before_moex_open(self) -> None:
        assert MOEX_MARKET_SCHEDULE.is_market_open(MOEX_BEFORE_OPEN_UTC) is False

    def test_is_market_closed_after_moex_close(self) -> None:
        assert MOEX_MARKET_SCHEDULE.is_market_open(MOEX_AFTER_CLOSE_UTC) is False

    def test_is_market_open_uses_current_time_when_dt_is_none(self) -> None:
        # Just verify it doesn't raise — result depends on real-time clock
        result = US_MARKET_SCHEDULE.is_market_open(None)
        assert isinstance(result, bool)

    def test_us_schedule_market_id(self) -> None:
        assert US_MARKET_SCHEDULE._market_id == "us"  # type: ignore[attr-defined]

    def test_moex_schedule_market_id(self) -> None:
        assert MOEX_MARKET_SCHEDULE._market_id == "moex"  # type: ignore[attr-defined]


# ── next_open ────────────────────────────────────────────────────────────


class TestNextOpen:
    def test_next_open_returns_future_datetime_before_open(self) -> None:
        next_open = US_MARKET_SCHEDULE.next_open(US_BEFORE_OPEN_UTC)
        # next open should be the same day at 09:30 ET = 14:30 UTC
        assert next_open > US_BEFORE_OPEN_UTC
        assert next_open.tzinfo is not None  # must be timezone-aware

    def test_next_open_returns_future_datetime_after_close(self) -> None:
        next_open = US_MARKET_SCHEDULE.next_open(US_AFTER_CLOSE_UTC)
        # next open should be the next trading day
        assert next_open > US_AFTER_CLOSE_UTC
        assert next_open.tzinfo is not None

    def test_next_open_is_utc_aware(self) -> None:
        next_open = US_MARKET_SCHEDULE.next_open(US_BEFORE_OPEN_UTC)
        # Result must be timezone-aware
        assert next_open.tzinfo is not None

    def test_next_open_uses_current_time_when_dt_is_none(self) -> None:
        # Verify it doesn't raise
        result = US_MARKET_SCHEDULE.next_open(None)
        assert result.tzinfo is not None

    def test_next_open_skips_weekend_to_monday(self) -> None:
        # Friday after close: next open must be Monday, not Saturday or Sunday
        next_open = US_MARKET_SCHEDULE.next_open(US_FRIDAY_AFTER_CLOSE_UTC)
        # Monday 2024-01-22 09:30 ET = 14:30 UTC
        expected_monday_open_utc = datetime(2024, 1, 22, 14, 30, tzinfo=UTC)
        assert next_open == expected_monday_open_utc


# ── SCHEDULES dict ───────────────────────────────────────────────────────


class TestSchedulesDict:
    def test_schedules_contains_us(self) -> None:
        assert "us" in SCHEDULES

    def test_schedules_contains_moex(self) -> None:
        assert "moex" in SCHEDULES

    def test_schedules_count(self) -> None:
        assert len(SCHEDULES) == EXPECTED_SCHEDULE_COUNT

    def test_schedules_us_is_correct_type(self) -> None:
        assert isinstance(SCHEDULES["us"], MarketSchedule)

    def test_schedules_moex_is_correct_type(self) -> None:
        assert isinstance(SCHEDULES["moex"], MarketSchedule)
