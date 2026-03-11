"""Tests for MOEX trading calendar."""

from __future__ import annotations

from datetime import date

import pytest

from finalayze.data.moex_calendar import is_moex_holiday, trading_days_gap


class TestIsHoliday:
    def test_new_year_jan_1(self) -> None:
        assert is_moex_holiday(date(2024, 1, 1)) is True

    def test_new_year_jan_8(self) -> None:
        assert is_moex_holiday(date(2024, 1, 8)) is True

    def test_defenders_day_feb_23(self) -> None:
        assert is_moex_holiday(date(2024, 2, 23)) is True

    def test_womens_day_mar_8(self) -> None:
        assert is_moex_holiday(date(2024, 3, 8)) is True

    def test_labour_day_may_1(self) -> None:
        assert is_moex_holiday(date(2024, 5, 1)) is True

    def test_victory_day_may_9(self) -> None:
        assert is_moex_holiday(date(2024, 5, 9)) is True

    def test_russia_day_jun_12(self) -> None:
        assert is_moex_holiday(date(2024, 6, 12)) is True

    def test_unity_day_nov_4(self) -> None:
        assert is_moex_holiday(date(2024, 11, 4)) is True

    def test_regular_trading_day(self) -> None:
        assert is_moex_holiday(date(2024, 3, 15)) is False  # Regular Friday

    def test_weekend_not_holiday(self) -> None:
        # Weekends handled separately — is_moex_holiday only checks public holidays
        # Saturday — weekend, not a public holiday
        assert is_moex_holiday(date(2024, 3, 16)) is False

    def test_jan_2_to_7_also_holidays(self) -> None:
        for day in range(2, 8):
            assert is_moex_holiday(date(2024, 1, day)) is True


class TestTradingDaysGap:
    def test_same_day_is_zero(self) -> None:
        assert trading_days_gap(date(2024, 3, 15), date(2024, 3, 15)) == 0

    def test_single_weekend_gap(self) -> None:
        # Friday to Monday = 2 weekend days, 0 holidays
        gap = trading_days_gap(date(2024, 3, 15), date(2024, 3, 18))
        assert gap == 2

    def test_new_year_holiday_block(self) -> None:
        # Dec 31 to Jan 9 = 9 calendar days; Jan 1-8 are holidays (8 days)
        # Dec 31 is Mon, Jan 8 is Mon, Jan 9 is Tue
        gap = trading_days_gap(date(2024, 12, 31), date(2025, 1, 9))
        # Jan 1-8 are all holidays; Dec 31 and Jan 9 (endpoints) are not counted
        # Actually: days between Dec 31 and Jan 9 = Jan 1..8 inclusive = 8 days
        # All 8 are non-trading (1-8 are holidays)
        assert gap >= 8  # At least 8 non-trading days

    def test_regular_week_no_gap(self) -> None:
        # Mon to Fri = no non-trading days
        gap = trading_days_gap(date(2024, 3, 11), date(2024, 3, 15))
        assert gap == 0
