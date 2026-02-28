"""Unit tests for daily/weekly loss limit tracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.risk.loss_limits import LossLimitTracker

# ── Constants ─────────────────────────────────────────────────────────────

INITIAL_EQUITY = Decimal(100_000)
DAILY_LIMIT_PCT = 3.0
WEEKLY_LIMIT_PCT = 5.0
COOLDOWN_DAYS = 1

# Equity levels for test scenarios
EQUITY_AFTER_2PCT_LOSS = Decimal(98_000)
EQUITY_AFTER_3PCT_LOSS = Decimal(97_000)
EQUITY_AFTER_4PCT_LOSS = Decimal(96_000)
EQUITY_AFTER_5PCT_LOSS = Decimal(95_000)
EQUITY_AFTER_6PCT_LOSS = Decimal(94_000)

MONDAY = datetime(2025, 1, 6, tzinfo=UTC)  # A Monday
MONDAY_NOON = datetime(2025, 1, 6, 12, 0, tzinfo=UTC)
TUESDAY = MONDAY + timedelta(days=1)
WEDNESDAY = MONDAY + timedelta(days=2)
THURSDAY = MONDAY + timedelta(days=3)
FRIDAY = MONDAY + timedelta(days=4)
NEXT_MONDAY = MONDAY + timedelta(days=7)

# For multi-day cooldown tests
COOLDOWN_2_DAYS = 2
MONDAY_PLUS_1 = MONDAY + timedelta(days=1)
MONDAY_PLUS_2 = MONDAY + timedelta(days=2)
MONDAY_PLUS_3 = MONDAY + timedelta(days=3)


class TestLossLimitTracker:
    """Tests for LossLimitTracker."""

    def test_no_halt_within_limits(self) -> None:
        """Trading is allowed when losses are within limits."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # 2% loss is within 3% daily limit
        assert not tracker.is_halted(MONDAY, EQUITY_AFTER_2PCT_LOSS)

    def test_daily_limit_triggers_halt(self) -> None:
        """Trading halts when daily loss exceeds limit."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # 4% loss exceeds 3% daily limit
        assert tracker.is_halted(MONDAY, EQUITY_AFTER_4PCT_LOSS)

    def test_daily_limit_exact_boundary(self) -> None:
        """Trading halts at exactly the daily limit."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # Exactly 3% loss triggers halt
        assert tracker.is_halted(MONDAY, EQUITY_AFTER_3PCT_LOSS)

    def test_cooldown_persists_within_period(self) -> None:
        """Halt persists during cooldown even if equity recovers."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # Trigger halt at noon on Monday
        assert tracker.is_halted(MONDAY_NOON, EQUITY_AFTER_4PCT_LOSS)

        # Later on Monday, equity recovers — still halted due to cooldown
        # (cooldown_days=1 means halted_until = MONDAY_NOON + 1 day = TUESDAY noon)
        assert tracker.is_halted(MONDAY_NOON, INITIAL_EQUITY)

    def test_cooldown_expires_after_period(self) -> None:
        """Halt clears after cooldown period expires via reset_day."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # Trigger halt
        assert tracker.is_halted(MONDAY, EQUITY_AFTER_4PCT_LOSS)

        # Next day: reset_day clears cooldown (Tuesday >= Monday + 1 day)
        tracker.reset_day(TUESDAY, EQUITY_AFTER_4PCT_LOSS)
        # No new breach (equity == baseline), so not halted
        assert not tracker.is_halted(TUESDAY, EQUITY_AFTER_4PCT_LOSS)

    def test_multi_day_cooldown(self) -> None:
        """With 2-day cooldown, halt persists for 2 days."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_2_DAYS,
        )
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # Trigger halt on Monday
        assert tracker.is_halted(MONDAY, EQUITY_AFTER_4PCT_LOSS)

        # Tuesday: still within 2-day cooldown (halted_until = Monday + 2 = Wednesday)
        tracker.reset_day(MONDAY_PLUS_1, INITIAL_EQUITY)
        assert tracker.is_halted(MONDAY_PLUS_1, INITIAL_EQUITY)

        # Wednesday: cooldown expired
        tracker.reset_day(MONDAY_PLUS_2, INITIAL_EQUITY)
        assert not tracker.is_halted(MONDAY_PLUS_2, INITIAL_EQUITY)

    def test_weekly_limit_triggers_halt(self) -> None:
        """Trading halts when cumulative weekly loss exceeds limit."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        # Start of week
        tracker.reset_week(MONDAY, INITIAL_EQUITY)
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # 2% loss Monday (within daily limit)
        assert not tracker.is_halted(MONDAY, EQUITY_AFTER_2PCT_LOSS)

        # Tuesday: another day, equity starts at 98k
        tracker.reset_day(TUESDAY, EQUITY_AFTER_2PCT_LOSS)
        # Drop to 95k = 5% from week start (100k), triggers weekly limit
        assert tracker.is_halted(TUESDAY, EQUITY_AFTER_5PCT_LOSS)

    def test_weekly_reset_clears_weekly_baseline(self) -> None:
        """New week resets the weekly baseline."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_week(MONDAY, INITIAL_EQUITY)
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # 5% weekly loss triggers halt
        assert tracker.is_halted(MONDAY, EQUITY_AFTER_5PCT_LOSS)

        # New week resets (cooldown also expired: NEXT_MONDAY >= MONDAY + 1 day)
        tracker.reset_week(NEXT_MONDAY, EQUITY_AFTER_5PCT_LOSS)
        tracker.reset_day(NEXT_MONDAY, EQUITY_AFTER_5PCT_LOSS)

        # No longer halted (new baseline is 95k)
        assert not tracker.is_halted(NEXT_MONDAY, EQUITY_AFTER_5PCT_LOSS)

    def test_gains_do_not_trigger_halt(self) -> None:
        """Positive equity changes do not trigger halt."""
        tracker = LossLimitTracker(
            daily_loss_limit_pct=DAILY_LIMIT_PCT,
            weekly_loss_limit_pct=WEEKLY_LIMIT_PCT,
            cooldown_days=COOLDOWN_DAYS,
        )
        tracker.reset_week(MONDAY, INITIAL_EQUITY)
        tracker.reset_day(MONDAY, INITIAL_EQUITY)

        # 10% gain
        gain_equity = Decimal(110_000)
        assert not tracker.is_halted(MONDAY, gain_equity)

    def test_default_parameters(self) -> None:
        """Default parameters work without explicit values."""
        tracker = LossLimitTracker()
        tracker.reset_day(MONDAY, INITIAL_EQUITY)
        tracker.reset_week(MONDAY, INITIAL_EQUITY)

        # Small loss should not trigger halt with defaults
        assert not tracker.is_halted(MONDAY, EQUITY_AFTER_2PCT_LOSS)
