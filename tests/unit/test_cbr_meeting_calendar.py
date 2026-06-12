"""Unit tests for CBR meeting calendar and CPI publication date helpers.

Tests the static CBR_MEETINGS data, CPI_PUBLICATION_DATES, and the helper
functions: get_last_cbr_decision, get_next_cbr_meeting, days_to_next_cbr,
get_latest_published_cpi_month.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.data.fetchers.cbr import (
    CBR_MEETINGS,
    CPI_PUBLICATION_DATES,
    CBRMeeting,
    days_to_next_cbr,
    get_last_cbr_decision,
    get_latest_published_cpi_month,
    get_next_cbr_meeting,
    get_recent_cbr_decisions,
    get_yield_slope_bps,
    is_cutting_cycle,
)

# ── Constants (no magic numbers, ruff PLR2004) ──────────────────────────────

FIRST_MEETING_YEAR = 2022
LAST_MEETING_YEAR = 2026
MIN_MEETING_COUNT = 30  # we have 2022-2026, at least ~8 per year
EMERGENCY_RATE = Decimal("20.00")
RATE_21 = Decimal("21.00")
RATE_18 = Decimal("18.00")

# Mid-2024 test dates
MID_2024 = date(2024, 8, 1)
EXPECTED_LAST_DECISION_DATE_MID_2024 = date(2024, 7, 26)
EXPECTED_NEXT_MEETING_DATE_MID_2024 = date(2024, 9, 13)
EXPECTED_DAYS_TO_NEXT_MID_2024 = 43  # Aug 1 -> Sep 13

# Edge case dates
BEFORE_ALL_MEETINGS = date(2020, 1, 1)
AFTER_ALL_MEETINGS = date(2030, 1, 1)
ON_MEETING_DAY = date(2024, 7, 26)  # core meeting, hike to 18%

# CPI dates
CPI_CHECK_DATE_AFTER_JAN2025 = date(2025, 2, 15)  # after Jan 2025 CPI published (2025-02-12)
CPI_CHECK_DATE_BEFORE_JAN2025 = date(2025, 2, 10)  # before Jan 2025 CPI published
CPI_BEFORE_ALL = date(2023, 1, 1)  # before any CPI publication in our data
EXPECTED_CPI_MONTH_AFTER_JAN2025 = "2025-01"
EXPECTED_CPI_MONTH_BEFORE_JAN2025 = "2024-12"

# ── Verified realized 2025-2026 CBR easing path (R-C, official cbr.ru archive) ─
# Each entry: (decision date, action, rate_after). The committed calendar had six
# WRONG rate values and two unfilled meetings; the real path is 21% -> 14.50%
# (first cut 2025-06-06, terminal 14.50% on 2026-04-24). No magic numbers (PLR2004).
_FIRST_CUT = (date(2025, 6, 6), "cut", Decimal("20.00"))  # FIRST cut: 06-06, not 07-25
_CUT_2025_07 = (date(2025, 7, 25), "cut", Decimal("18.00"))
_CUT_2025_09 = (date(2025, 9, 12), "cut", Decimal("17.00"))
_CUT_2025_10 = (date(2025, 10, 24), "cut", Decimal("16.50"))
_CUT_2025_12 = (date(2025, 12, 19), "cut", Decimal("16.00"))
_CUT_2026_02 = (date(2026, 2, 13), "cut", Decimal("15.50"))
_CUT_2026_03 = (date(2026, 3, 20), "cut", Decimal("15.00"))  # was None (filled)
_CUT_2026_04 = (date(2026, 4, 24), "cut", Decimal("14.50"))  # was None (filled), terminal
_REALIZED_EASING_PATH = (
    _FIRST_CUT,
    _CUT_2025_07,
    _CUT_2025_09,
    _CUT_2025_10,
    _CUT_2025_12,
    _CUT_2026_02,
    _CUT_2026_03,
    _CUT_2026_04,
)

# The terminal realized rate is observed as-of the binding window endpoint.
_BINDING_WINDOW_END = date(2026, 6, 10)
_TERMINAL_REALIZED_RATE = Decimal("14.50")


# ── CBR_MEETINGS data integrity tests ───────────────────────────────────────


class TestCBRMeetingsData:
    """Verify structural integrity of the CBR_MEETINGS constant."""

    def test_is_nonempty_tuple(self) -> None:
        assert isinstance(CBR_MEETINGS, tuple)
        assert len(CBR_MEETINGS) >= MIN_MEETING_COUNT

    def test_all_entries_are_cbr_meeting(self) -> None:
        for m in CBR_MEETINGS:
            assert isinstance(m, CBRMeeting)

    def test_sorted_chronologically(self) -> None:
        dates = [m.date for m in CBR_MEETINGS]
        assert dates == sorted(dates)

    def test_year_range(self) -> None:
        assert CBR_MEETINGS[0].date.year == FIRST_MEETING_YEAR
        assert CBR_MEETINGS[-1].date.year == LAST_MEETING_YEAR

    def test_meeting_types_valid(self) -> None:
        valid_types = {"core", "interim", "emergency"}
        for m in CBR_MEETINGS:
            assert m.meeting_type in valid_types, f"Invalid type: {m.meeting_type} on {m.date}"

    def test_decisions_valid(self) -> None:
        valid_decisions = {"cut", "hold", "hike", None}
        for m in CBR_MEETINGS:
            assert m.decision in valid_decisions, f"Invalid decision: {m.decision} on {m.date}"

    def test_rate_after_present_when_decision_present(self) -> None:
        for m in CBR_MEETINGS:
            if m.decision is not None:
                assert m.rate_after is not None, f"Missing rate_after on {m.date}"
            else:
                assert m.rate_after is None, f"Unexpected rate_after on future meeting {m.date}"

    def test_emergency_feb_2022(self) -> None:
        """The February 2022 emergency hike to 20% is a known anchor point."""
        feb2022 = [m for m in CBR_MEETINGS if m.date == date(2022, 2, 28)]
        assert len(feb2022) == 1
        assert feb2022[0].meeting_type == "emergency"
        assert feb2022[0].decision == "hike"
        assert feb2022[0].rate_after == EMERGENCY_RATE

    def test_frozen_dataclass(self) -> None:
        """CBRMeeting should be immutable (frozen=True)."""
        m = CBR_MEETINGS[0]
        with pytest.raises(AttributeError):
            m.decision = "hold"  # type: ignore[misc]


# ── Realized 2025-2026 easing-path value spot-check (R-C / D-03) ─────────────


def test_realized_2025_2026_path_matches_cbr_archive() -> None:
    """The committed 2025-2026 cuts match the verified cbr.ru realized path (R-C).

    Spot-checks each decision date against the official archive: first cut
    2025-06-06 -> 20.00 (NOT a hold at 21.00), terminal 2026-04-24 -> 14.50.
    RED before Task 2: the un-corrected calendar carries 21/20/19/18/17/16 + two
    None, so this assertion fails until CBR_MEETINGS is corrected.
    """
    by_date = {m.date: m for m in CBR_MEETINGS}
    for decision_date, action, rate_after in _REALIZED_EASING_PATH:
        meeting = by_date.get(decision_date)
        assert meeting is not None, f"No CBR meeting on {decision_date}"
        assert meeting.decision == action, (
            f"{decision_date}: expected action {action!r}, got {meeting.decision!r}"
        )
        assert meeting.rate_after == rate_after, (
            f"{decision_date}: expected rate {rate_after}, got {meeting.rate_after}"
        )


def test_terminal_realized_rate_is_1450_at_binding_end() -> None:
    """The terminal realized key rate at the binding window end is 14.50% (R-C).

    The path ended at 14.50% on 2026-04-24; the 2026-06-19 meeting is still
    future as-of 2026-06-10, so the last decided rate is the terminal one.
    """
    last = get_last_cbr_decision(_BINDING_WINDOW_END)
    assert last is not None
    assert last.rate_after == _TERMINAL_REALIZED_RATE


# ── CPI_PUBLICATION_DATES data integrity tests ──────────────────────────────


class TestCPIPublicationDates:
    """Verify structural integrity of CPI_PUBLICATION_DATES."""

    def test_is_nonempty_dict(self) -> None:
        assert isinstance(CPI_PUBLICATION_DATES, dict)
        assert len(CPI_PUBLICATION_DATES) > 0

    def test_keys_are_yyyy_mm_format(self) -> None:
        import re

        pattern = re.compile(r"^\d{4}-\d{2}$")
        for key in CPI_PUBLICATION_DATES:
            assert pattern.match(key), f"Invalid key format: {key}"

    def test_values_are_dates(self) -> None:
        for key, val in CPI_PUBLICATION_DATES.items():
            assert isinstance(val, date), f"Value for {key} is not a date"

    def test_publication_after_covered_month(self) -> None:
        """Publication date must be after the month it covers."""
        for month_str, pub_date in CPI_PUBLICATION_DATES.items():
            year, month = month_str.split("-")
            # The publication must be after the end of the covered month
            last_day_of_month = date(int(year), int(month), 28)  # conservative
            assert pub_date > last_day_of_month, (
                f"CPI for {month_str} published on {pub_date} which is within the month"
            )


# ── get_last_cbr_decision tests ─────────────────────────────────────────────


class TestGetLastCBRDecision:
    def test_mid_2024_returns_july_meeting(self) -> None:
        result = get_last_cbr_decision(MID_2024)
        assert result is not None
        assert result.date == EXPECTED_LAST_DECISION_DATE_MID_2024
        assert result.decision == "hike"
        assert result.rate_after == RATE_18

    def test_on_meeting_day_includes_that_meeting(self) -> None:
        result = get_last_cbr_decision(ON_MEETING_DAY)
        assert result is not None
        assert result.date == ON_MEETING_DAY

    def test_before_all_meetings_returns_none(self) -> None:
        result = get_last_cbr_decision(BEFORE_ALL_MEETINGS)
        assert result is None

    def test_after_all_meetings_returns_last_with_decision(self) -> None:
        result = get_last_cbr_decision(AFTER_ALL_MEETINGS)
        assert result is not None
        # Should be the last meeting that has a non-None decision
        decided = [m for m in CBR_MEETINGS if m.decision is not None]
        assert result == decided[-1]

    def test_skips_future_meetings_without_decision(self) -> None:
        """Meetings with decision=None should not be returned even if date <= as_of."""
        # Find a future meeting without decision
        future_no_decision = [m for m in CBR_MEETINGS if m.decision is None]
        if future_no_decision:
            # Use a date on that meeting day
            as_of = future_no_decision[0].date
            result = get_last_cbr_decision(as_of)
            # Should NOT return the meeting without a decision
            assert result is not None
            assert result.decision is not None


# ── get_next_cbr_meeting tests ──────────────────────────────────────────────


class TestGetNextCBRMeeting:
    def test_mid_2024_returns_september_meeting(self) -> None:
        result = get_next_cbr_meeting(MID_2024)
        assert result is not None
        assert result.date == EXPECTED_NEXT_MEETING_DATE_MID_2024

    def test_on_meeting_day_returns_next_not_same(self) -> None:
        """get_next returns strictly after as_of, not the same day."""
        result = get_next_cbr_meeting(ON_MEETING_DAY)
        assert result is not None
        assert result.date > ON_MEETING_DAY

    def test_before_all_meetings_returns_first(self) -> None:
        result = get_next_cbr_meeting(BEFORE_ALL_MEETINGS)
        assert result is not None
        assert result == CBR_MEETINGS[0]

    def test_after_all_meetings_returns_none(self) -> None:
        result = get_next_cbr_meeting(AFTER_ALL_MEETINGS)
        assert result is None


# ── days_to_next_cbr tests ──────────────────────────────────────────────────


class TestDaysToNextCBR:
    def test_mid_2024_correct_count(self) -> None:
        result = days_to_next_cbr(MID_2024)
        assert result == EXPECTED_DAYS_TO_NEXT_MID_2024

    def test_returns_positive_integer(self) -> None:
        result = days_to_next_cbr(MID_2024)
        assert result is not None
        assert result > 0

    def test_after_all_meetings_returns_none(self) -> None:
        result = days_to_next_cbr(AFTER_ALL_MEETINGS)
        assert result is None

    def test_day_before_meeting_returns_one(self) -> None:
        first_meeting = CBR_MEETINGS[0]
        day_before = date(
            first_meeting.date.year,
            first_meeting.date.month,
            first_meeting.date.day - 1,
        )
        result = days_to_next_cbr(day_before)
        assert result == 1


# ── get_latest_published_cpi_month tests ────────────────────────────────────


class TestGetLatestPublishedCPIMonth:
    def test_after_jan2025_publication(self) -> None:
        """After Feb 12 2025, Jan 2025 CPI should be available."""
        result = get_latest_published_cpi_month(CPI_CHECK_DATE_AFTER_JAN2025)
        assert result == EXPECTED_CPI_MONTH_AFTER_JAN2025

    def test_before_jan2025_publication(self) -> None:
        """Before Feb 12 2025, Jan 2025 CPI should NOT be available yet."""
        result = get_latest_published_cpi_month(CPI_CHECK_DATE_BEFORE_JAN2025)
        assert result == EXPECTED_CPI_MONTH_BEFORE_JAN2025

    def test_no_lookahead(self) -> None:
        """The returned month's publication date must be <= as_of."""
        as_of = date(2025, 6, 1)
        result = get_latest_published_cpi_month(as_of)
        assert result is not None
        pub_date = CPI_PUBLICATION_DATES[result]
        assert pub_date <= as_of

    def test_before_all_publications_returns_none(self) -> None:
        result = get_latest_published_cpi_month(CPI_BEFORE_ALL)
        assert result is None

    def test_returns_yyyy_mm_string(self) -> None:
        result = get_latest_published_cpi_month(date(2025, 12, 31))
        assert result is not None
        assert len(result.split("-")) == 2  # noqa: PLR2004


# ── get_recent_cbr_decisions tests ─────────────────────────────────────────


class TestGetRecentCBRDecisions:
    """Test get_recent_cbr_decisions helper."""

    def test_two_cuts_oct_2025(self) -> None:
        """Oct 30 2025: last 2 decisions are 2025-10-24 (cut) and 2025-09-12 (cut)."""
        result = get_recent_cbr_decisions(date(2025, 10, 30), count=2)
        assert result == ["cut", "cut"]

    def test_two_holds_mid_2023(self) -> None:
        """Mid-2023: the holding cycle before July 2023 hike."""
        result = get_recent_cbr_decisions(date(2023, 6, 15), count=2)
        assert result == ["hold", "hold"]

    def test_before_all_meetings_returns_empty(self) -> None:
        result = get_recent_cbr_decisions(date(2020, 1, 1), count=2)
        assert result == []

    def test_count_three_default(self) -> None:
        """Default count=3 returns 3 decisions."""
        result = get_recent_cbr_decisions(date(2025, 12, 31))
        assert len(result) == 3  # noqa: PLR2004


# ── is_cutting_cycle tests ─────────────────────────────────────────────────


class TestIsCuttingCycle:
    """Test is_cutting_cycle helper."""

    def test_cutting_cycle_oct_2025(self) -> None:
        """Two consecutive cuts by Oct 30 2025 -> True."""
        assert is_cutting_cycle(date(2025, 10, 30)) is True

    def test_cutting_cycle_jul_2025_two_cuts(self) -> None:
        """Jul 30 2025: two cuts so far (2025-06-06, 2025-07-25) -> True (R-C).

        The corrected realized calendar makes 2025-06-06 the FIRST cut (was a
        spurious 21.00 hold), so by Jul 30 2025 the last two decisions are both
        cuts -- a cutting cycle. (Pre-correction this date had only one cut.)
        """
        assert is_cutting_cycle(date(2025, 7, 30)) is True

    def test_not_cutting_after_first_cut_jun_2025(self) -> None:
        """Jun 30 2025: only the 2025-06-06 cut so far, before it were holds -> False."""
        assert is_cutting_cycle(date(2025, 6, 30)) is False

    def test_not_cutting_holds_mid_2023(self) -> None:
        """Mid-2023 holds -> not cutting."""
        assert is_cutting_cycle(date(2023, 6, 15)) is False

    def test_not_cutting_before_all_meetings(self) -> None:
        """Before all meetings -> False (no decisions)."""
        assert is_cutting_cycle(date(2020, 1, 1)) is False


# ── get_yield_slope_bps tests ──────────────────────────────────────────────


YIELD_SLOPE_JUN_2024 = -150.0
YIELD_SLOPE_MAR_2022 = -250.0


class TestGetYieldSlopeBps:
    """Test get_yield_slope_bps helper."""

    def test_mid_2024(self) -> None:
        """2024-06-15 -> nearest month 2024-06 = -150.0 bps."""
        assert get_yield_slope_bps(date(2024, 6, 15)) == YIELD_SLOPE_JUN_2024

    def test_early_2022(self) -> None:
        """2022-03-15 -> nearest month 2022-03 = -250.0 bps."""
        assert get_yield_slope_bps(date(2022, 3, 15)) == YIELD_SLOPE_MAR_2022

    def test_no_data_returns_zero(self) -> None:
        """2021-01-01 -> no data -> 0.0."""
        assert get_yield_slope_bps(date(2021, 1, 1)) == 0.0

    def test_between_months_uses_earlier(self) -> None:
        """2022-05-15 -> 2022-04 is the latest key <= 2022-05."""
        result = get_yield_slope_bps(date(2022, 5, 15))
        expected = -180.0  # 2022-04 value
        assert result == expected
